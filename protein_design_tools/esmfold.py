import pandas
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from pathlib import Path
import torch, gc
import sys

def convert_outputs_to_pdb(outputs):
    """
        take esm outputs and return an actual PDB file
    """
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").detach().numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().detach().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def esmfold(seq_output_df, model_name="facebook/esmfold_v1", num_recycles=5, device="cuda", gpu="A100"):
    """
    Fold protein sequences using the ESMFold model and save the predicted structures.

    Args:
        seq_output_df (pd.DataFrame): DataFrame containing "sequence" and "output_path" columns.
        model_name (str, optional): Name of the pretrained ESMFold model. Defaults to "facebook/esmfold_v1".
        num_recycles (int, optional): Number of recycles during folding. Defaults to 5.
        device (str, optional): Device to run the model on ("cuda" or "cpu"). Defaults to "cuda".
        gpu (str, optional): GPU type for performance optimizations (e.g., "A100"). Defaults to "A100".

    Returns:
        None
    """

    # Load the model 
    if device == "cuda":
        # House-keeping to free up memory
        gc.collect()
        torch.cuda.empty_cache()

        model = EsmForProteinFolding.from_pretrained(model_name)
        model = model.cuda()
    elif device == "cpu":
        model = EsmForProteinFolding.from_pretrained(model_name)
        model = model.cpu()

    # Tokenize the sequence
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Performance improvements for different GPUs. If there is demand I'll add other GPUs
    if gpu == "A100":
        model.esm = model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True
        model.trunk.set_chunk_size(64)

    for index, row in seq_output_df.iterrows():
        # Get the sequence and the output path
        sequence = row["sequence"]
        output_pdb = row["output_path"]

        # check if output_pdb.parent exists, and if not create it
        output_pdb = Path(output_pdb)
        if not output_pdb.parent.exists():
            output_pdb.parent.mkdir(parents=True, exist_ok=True)

        # check if output_pdb exists, if it does, skip
        if output_pdb.exists():
            print(f"Output file {output_pdb} already exists, skipping...")
            continue
        else:
            try:
                tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
                tokenized_input = tokenized_input.cuda()
                with torch.no_grad():
                    output = model(tokenized_input, num_recycles=5)

                # Convert the output to a PDB file
                pdbs = convert_outputs_to_pdb(output)
                with open(output_pdb, "w") as f:
                    f.write(pdbs[0])
            except:
                print("\nError processing sequence")
                print(f"  Sequence: {sequence}")
                print(f"  Output PDB: {output_pdb}")
                print(f"\nPerforming garbage collection and emptying cache...")
                gc.collect()
                torch.cuda.empty_cache()

                # load model and send to GPU
                model = EsmForProteinFolding.from_pretrained(model_name)
                model = model.cuda()

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                try:
                    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids']
                    tokenized_input = tokenized_input.cuda()
                    with torch.no_grad():
                        output = model(tokenized_input, num_recycles)

                    # Convert the output to a PDB file
                    pdbs = convert_outputs_to_pdb(output)
                    with open(output_pdb, "w") as f:
                        f.write(pdbs[0])
                except:
                    print(f"Error processing sequence {sequence} again. Skipping...")
                    continue
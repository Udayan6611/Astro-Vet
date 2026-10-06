import lightkurve as lk
import pandas as pd
import os
import time

# 1 = Confirmed Exoplanet, 0 = False Positive (Eclipsing Binary / Blends)
SANDBOX_TARGETS = {
    "261136679": 1,
    "150428135": 1,
    "279741379": 1,
    "281541555": 1,
    "52368076":  1,
    "320004517": 0,  # Replaced the missing SPOC target
    "149603524": 0,
    "307210830": 0,
    "425933644": 0,
    "207141131": 0
}

def build_sandbox(out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    
    csv_path = os.path.join(out_dir, "sandbox_labels.csv")
    
    # Load existing records to avoid re-downloading
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        records = existing_df.to_dict('records')
        downloaded_tics = [str(r['tic_id']) for r in records]
    else:
        downloaded_tics = []

    print("[*] Starting Sandbox Dataset Download...")
    
    for tic_id, label in SANDBOX_TARGETS.items():
        if str(tic_id) in downloaded_tics:
            print(f" -> TIC {tic_id} already downloaded. Skipping.")
            continue

        print(f" -> Fetching TIC {tic_id}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                search = lk.search_targetpixelfile(f"TIC {tic_id}", author="SPOC")
                if len(search) > 0:
                    tpf = search[0].download(download_dir=out_dir, quality_bitmask='hard')
                    if tpf is not None:
                        records.append({
                            "tic_id": tic_id,
                            "file_path": tpf.path,
                            "label": label
                        })
                        print(f"    [+] Success.")
                        break  # Break out of retry loop on success
                else:
                    print(f"    [-] No SPOC data found.")
                    break
            except Exception as e:
                print(f"    [-] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"    [*] Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"    [-] Max retries reached for TIC {tic_id}. Moving to next.")
            
        # Save incrementally after every target in case of another crash
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        
    print(f"\n[+] Sandbox build complete! Total TPFs: {len(records)}.")
    print(f"[+] Labels saved to {csv_path}")

if __name__ == "__main__":
    build_sandbox()
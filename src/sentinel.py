import lightkurve as lk
import matplotlib.pyplot as plt
import os

def run_sentinel(tic_id, sector, out_dir="data"):
    print(f"[*] Querying MAST API for TIC {tic_id} (Sector {sector})...")
    
    # 1. Search and Download the Target Pixel File (TPF)
    search_result = lk.search_targetpixelfile(f"TIC {tic_id}", sector=sector, author="SPOC")
    tpf = search_result.download(download_dir=out_dir)
    
    if tpf is None:
        print("[-] Failed to download TPF.")
        return

    print(f"[+] Downloaded TPF: {tpf.shape[0]} cadences (frames) found.")

    # 2. Clean the Data (The Sentinel Bitmask)
    # 'hard' removes all cadences flagged for anomalies by NASA
    tpf_clean = tpf[tpf.quality_mask == 'hard']
    print(f"[+] Cleaned Data: {tpf_clean.shape[0]} valid cadences remain.")

    # 3. Extract and Normalize the Light Curve
    # Aperture photometry extracts the 1D light curve from the 2D pixels
    lc = tpf_clean.to_lightcurve(aperture_mask=tpf_clean.pipeline_mask)
    lc_clean = lc.remove_nans().normalize().flatten(window_length=101)

    # 4. Generate Output Visuals for Review 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: The 2D Pixel Heatmap (Cadence 0)
    tpf_clean.plot(ax=axes[0], frame=0, aperture_mask=tpf_clean.pipeline_mask, title="TPF Pixel Heatmap")
    
    # Plot 2: The 1D Cleaned Light Curve
    lc_clean.plot(ax=axes[1], title="Normalized Light Curve (Sentinel Output)")
    
    plt.tight_layout()
    
    # Save the plot for the project report
    os.makedirs("notebooks", exist_ok=True)
    save_path = f"notebooks/TIC_{tic_id}_Sentinel_Output.png"
    plt.savefig(save_path)
    print(f"[+] Visuals saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Test with Pi Mensae (A known TESS exoplanet host)
    run_sentinel(tic_id="261136679", sector=1)
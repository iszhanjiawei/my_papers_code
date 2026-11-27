"""Run the real-parent REPA check with the historical adaptive-band configuration."""

from aligndit.script.misc.smoke_test_foley_fixed_band_repa_real_parent import main


if __name__ == "__main__":
    main(default_config="finetune_celebvdub_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus")

paper/figures/  — every PNG used by main.tex in one directory (easy zip/upload)

Embedded in the PDF:
  fixed_loss_curves.png
  fixed_p_hat_tracking.png
  phase2_alpha1.5.png
  phase3_halfcheetah.png

Also bundled (caption mentions):
  phase3_hopper.png
  phase3_walker2d.png

Source: ../results/phase{1,2,3}/figures/ after running experiments.

Refresh from repo root:
  cp results/phase1/figures/fixed_loss_curves.png results/phase1/figures/fixed_p_hat_tracking.png paper/figures/
  cp results/phase2/figures/phase2_alpha1.5.png paper/figures/
  cp results/phase3/figures/phase3_halfcheetah.png results/phase3/figures/phase3_hopper.png results/phase3/figures/phase3_walker2d.png paper/figures/

Build: cd ../ && pdflatex main.tex && pdflatex main.tex

main.tex: \graphicspath{{figures/}}

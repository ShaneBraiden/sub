# Coarse-sync recovery (test)

| File | Method | Split injected | Drift | Offset (s) | abs Δa | abs Δb (s) |
|---|---|---|---|---|---|---|
| libri-test-clean-1284-1180 | ransac | no | 1.0000 | -2.48 | 0.00062 | 0.311 |
| libri-test-clean-260-123288 | ransac | no | 1.0000 | -1.71 | 0.00044 | 0.424 |
| libri-test-clean-2961-960 | piecewise | yes | 1.0000 | +1.00 | 0.00000 | 0.409 |
| libri-test-clean-3575-170457 | ransac | no | 1.0000 | +2.13 | 0.00069 | 0.189 |
| libri-test-clean-4446-2275 | ransac | no | 1.0427 | +2.07 | 0.00070 | 0.235 |
| libri-test-clean-5142-33396 | ransac | no | 1.0000 | -0.78 | 0.00177 | 0.236 |
| libri-test-clean-6829-68769 | ransac | no | 1.0000 | +3.22 | 0.00129 | 0.368 |
| libri-test-clean-8463-294828 | ransac | no | 1.0000 | +1.34 | 0.00108 | 0.204 |
| libri-test-other-1998-15444 | ransac | no | 1.0000 | +2.51 | 0.00097 | 0.388 |
| libri-test-other-2414-159411 | piecewise | yes | 1.0000 | +3.57 | 0.00147 | 0.217 |
| libri-test-other-3764-168670 | ransac | no | 1.0000 | +2.84 | 0.00069 | 0.484 |
| libri-test-other-4294-35475 | ransac | no | 0.9590 | -0.36 | 0.00191 | 0.550 |
| libri-test-other-4350-9170 | ransac | no | 1.0000 | +2.33 | 0.00073 | 0.191 |
| libri-test-other-6070-86744 | ransac | no | 0.9590 | -1.08 | 0.00018 | 0.353 |
| libri-test-other-6128-63240 | ransac | no | 1.0000 | -0.11 | 0.00276 | 0.006 |
| libri-test-other-8461-281231 | ransac | no | 1.0000 | -0.34 | 0.00183 | 0.719 |
| ted-test-AimeeMullins | ransac | no | 1.0000 | +0.68 | 0.00043 | 0.150 |
| ted-test-BillGates | piecewise | yes | 1.0427 | +2.65 | 0.00000 | 0.160 |
| ted-test-DanBarber | piecewise | yes | 1.0000 | +0.43 | 0.00000 | 0.173 |
| ted-test-DanBarber_2010_S103 | ransac | no | 1.0000 | -4.25 | 0.00000 | 2.265 |
| ted-test-DanielKahneman | ransac | no | 0.9590 | +4.30 | 0.00014 | 0.089 |
| ted-test-EricMead_2009P_EricMead | ransac | no | 1.0000 | +1.48 | 0.00047 | 0.074 |
| ted-test-GaryFlake | ransac | no | 0.9590 | +0.08 | 0.00044 | 0.136 |
| ted-test-JamesCameron | ransac | no | 0.9590 | -0.06 | 0.00008 | 0.233 |
| ted-test-JaneMcGonigal | ransac | no | 1.0000 | -4.40 | 0.00007 | 0.119 |
| ted-test-MichaelSpecter | ransac | no | 1.0000 | -0.26 | 0.00024 | 0.277 |
| ted-test-RobertGupta | ransac | no | 1.0000 | -4.67 | 0.00003 | 0.241 |

Summary: {"files": 27, "median_abs_da": 0.0004726344281732464, "max_abs_da": 0.0027643595356871, "median_abs_db_s": 0.23452724178682827, "share_db_below_250ms": 0.5925925925925926, "share_db_below_500ms": 0.8888888888888888, "splits_injected": 4, "splits_detected": 4, "false_piecewise": 0}

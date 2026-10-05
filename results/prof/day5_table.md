| Run | Main kernel name | Calls | Average duration (µs) | Harness time (µs) | Harness vs kernel |
|---|---|---|---|---|---|
| rocblas_4096 | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_..._MT256x128x32_MI32x32x1` | 60 | 1313.19 | 1359.38 | +3.5% |
| triton_4096 | `gemm_kernel_v2` | 1714 | 1350.21 | 1377.46 | +2.0% |
| rocblas_prefill | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_..._MT256x208x32_MI16x16x1` | 60 | 1618.73 | 1661.30 | +2.6% |
| triton_prefill | `gemm_kernel_v2` | 1350 | 1820.08 | 1847.07 | +1.5% |
| rocblas_decode | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_..._MT64x16x64_MI16x16x1` | 60 | 40.53 | 48.48 | +19.6% |
| triton_decode | `gemm_kernel_v2` | 7946 | 63.18 | 74.24 | +17.5% |

Notes:
- **rocblas_4096** full kernel name: `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x128x32_MI32x32x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA90a_IU1_K1_LBSPPA0_LBSPPB128_LBSPPM0_LPA0_LPB8_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_2_MO1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB1_WSGRA0_WSGRB1_WS64_WG64_4_1`
- **triton_4096** full kernel name: `gemm_kernel_v2`
  - kernel_stats average is 1700.79 µs over all 1714 calls; the table uses the last 50 calls (1350.21 µs) because warm-up differs
- **rocblas_prefill** full kernel name: `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT256x208x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA2_GRVWB2_GSUAMB_GLS0_ISA90a_IU1_K1_LBSPPA2048_LBSPPB128_LBSPPM0_LPA16_LPB4_LPM0_LRVW4_LWPMn1_MIAV0_MIWT4_13_MO1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB1_WSGRA0_WSGRB1_WS64_WG64_4_1`
- **triton_prefill** full kernel name: `gemm_kernel_v2`
  - kernel_stats average is 2293.75 µs over all 1350 calls; the table uses the last 50 calls (1820.08 µs) because warm-up differs
- **rocblas_decode** full kernel name: `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GRVWB4_GSUAMB_GLS0_ISA90a_IU1_K1_LBSPPA1024_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS2_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_SPO1_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB1_WS64_WG64_4_1`
- **triton_decode** full kernel name: `gemm_kernel_v2`
  - kernel_stats average is 155.82 µs over all 7946 calls; the table uses the last 50 calls (63.18 µs) because warm-up differs

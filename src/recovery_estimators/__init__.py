"""Estimators for the quantity the readout consumes.

The recovery map's bound is stated through `J_Delta`, the error of the contrast field
`w^(l) = s_hat(.,u^l) - s_hat(.,u^0)` under the clean marginal. Nothing in the theorem
constrains how `s_hat` is obtained, so which objective is used is an open choice, and this
package exists to compare choices on one readout pipeline.
"""

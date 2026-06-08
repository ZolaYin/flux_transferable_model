"""
Config entry point for FluxHierarchicalMoEModel (--dataset carbonbench_flux_hiermoe).
Delegates to get_carbonbench_flux_hiermoe_config() in the base config module.
"""
from .carbonbench_flux_scratch_config import get_carbonbench_flux_hiermoe_config


def get_carbonbench_flux_hiermoe_from_scratch_config():
    return get_carbonbench_flux_hiermoe_config()

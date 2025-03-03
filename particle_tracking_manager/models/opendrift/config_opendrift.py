import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict

from pydantic import BaseModel, Field, computed_field, model_validator
from typing_extensions import Self


logger = logging.getLogger()




# Enum for drift_model
class DriftModelEnum(str, Enum):
    OceanDrift = "OceanDrift"
    LarvalFish = "LarvalFish"
    OpenOil = "OpenOil"
    Leeway = "Leeway"


# Enum for radius_type
class RadiusTypeEnum(str, Enum):
    gaussian = "gaussian"
    uniform = "uniform"


# class OpenDriftConfig(TheManagerConfig):
class OpenDriftConfig(BaseModel):
    # input from TheManagerConfig
    use_cache: bool
    stokes_drift: bool
    do3D: bool
    wind_drift_factor: Optional[float]
    use_static_masks: bool
    vertical_mixing: bool
    model_drop_vars: List[str]
    ocean_model: str

    interpolator_filename: Optional[str] = Field(None, description="Filename to save interpolator to.", ptm_level=3)


    
    drift_model: DriftModelEnum = Field(
        default=DriftModelEnum.OceanDrift,
        description="Which model in OpenDrift to use. This corresponds to the type of drift scenario the user wants to run.",
        ptm_level=1,
    )
    
    export_variables: Optional[List[str]] = Field(
        default=None,
        description="List of variables to export. Options available with `m.all_export_variables` for a given `drift_model`. "
                    "['lon', 'lat', 'ID', 'status', 'z'] will always be exported. Default of None means all possible variables are exported.",
        ptm_level=3,
    )
    
    max_speed: float = Field(default=5.0, od_mapping="drift:max_speed", description="Maximum drift speed.", ptm_level=1)
    
    object_type: str = Field(
        default="Person-in-water (PIW), unknown state (mean values)", 
        od_mapping="seed:object_type", 
        ptm_level=1,
        description="The object type associated with the drift model."
    )
    
    diameter: float = Field(default=0.0014, od_mapping="seed:diameter", ptm_level=2)
    
    neutral_buoyancy_salinity: float = Field(default=31.25, od_mapping="seed:neutral_buoyancy_salinity", ptm_level=2)
    
    stage_fraction: float = Field(default=0.0, od_mapping="seed:stage_fraction", ptm_level=2)
    
    hatched: int = Field(default=0, od_mapping="seed:hatched", ptm_level=2)
    
    length: float = Field(default=0.0, od_mapping="seed:length", ptm_level=2)
    
    weight: float = Field(default=0.08, od_mapping="seed:weight", ptm_level=2)
    
    oil_type: str = Field(default="GENERIC MEDIUM CRUDE", od_mapping="seed:oil_type", ptm_level=1)
    
    m3_per_hour: float = Field(default=1.0, od_mapping="seed:m3_per_hour", ptm_level=2)
    
    oil_film_thickness: float = Field(default=1.0, od_mapping="seed:oil_film_thickness", ptm_level=3)
    
    droplet_size_distribution: str = Field(default="uniform", od_mapping="seed:droplet_size_distribution", ptm_level=3)
    
    droplet_diameter_mu: float = Field(default=0.001, od_mapping="seed:droplet_diameter_mu", ptm_level=3)
    
    droplet_diameter_sigma: float = Field(default=0.0005, od_mapping="seed:droplet_diameter_sigma", ptm_level=3)
    
    droplet_diameter_min_subsea: float = Field(default=0.0005, od_mapping="seed:droplet_diameter_min_subsea", ptm_level=3)
    
    droplet_diameter_max_subsea: float = Field(default=0.005, od_mapping="seed:droplet_diameter_max_subsea", ptm_level=3)
    
    emulsification: bool = Field(default=True, od_mapping="processes:emulsification", ptm_level=2)
    
    dispersion: bool = Field(default=True, od_mapping="processes:dispersion", ptm_level=2)
    
    evaporation: bool = Field(default=True, od_mapping="processes:evaporation", ptm_level=2)
    
    update_oilfilm_thickness: bool = Field(default=True, od_mapping="processes:update_oilfilm_thickness", ptm_level=2)
    
    biodegradation: bool = Field(default=True, od_mapping="processes:biodegradation", ptm_level=2)
    
    plots: Optional[Dict[str, str]] = Field(default=None, ptm_level=1, description="Dictionary of plots to generate using OpenDrift.")
    
    radius: float = Field(default=1000.0, ptm_level=2, min=0.0, max=1000000, units="m", description="Radius around each lon-lat pair, within which particles will be randomly seeded.")
    
    radius_type: RadiusTypeEnum = Field(default=RadiusTypeEnum.gaussian, ptm_level=3, description="Radius type. Options: 'gaussian' or 'uniform'.")
    
    diffusivitymodel: str = Field(default="windspeed_Large1994", od_mapping="vertical_mixing:diffusivitymodel", ptm_level=3)
    
    use_auto_landmask: bool = Field(default=False, od_mapping="general:use_auto_landmask", ptm_level=3)
    
    mixed_layer_depth: float = Field(default=30.0, od_mapping="environment:fallback:ocean_mixed_layer_thickness", ptm_level=3)
    
    coastline_action: str = Field(default="previous", od_mapping="general:coastline_action", ptm_level=2)
    
    seafloor_action: str = Field(default="previous", od_mapping="general:seafloor_action", ptm_level=2)
    
    current_uncertainty: float = Field(default=0.0, value=0.0, od_mapping="drift:current_uncertainty", ptm_level=2)
    
    wind_uncertainty: float = Field(default=0.0, value=0.0, od_mapping="drift:wind_uncertainty", ptm_level=2)
    
    wind_drift_depth: float = Field(default=0.02, od_mapping="drift:wind_drift_depth", ptm_level=3)
    
    vertical_mixing_timestep: int = Field(default=60, od_mapping="vertical_mixing:timestep", ptm_level=3)
    
    save_interpolator: bool = Field(default=False)

    @model_validator(mode='after')
    def check_interpolator_filename(self) -> Self:
        if self.interpolator_filename is not None and not self.use_cache:
            raise ValueError("If interpolator_filename is input, use_cache must be True.")
        return self
    
    
    @model_validator(mode='after')
    def setup_interpolator(self) -> Self:
        """Setup interpolator."""

        if self.use_cache:
            if self.interpolator_filename is None:
                # TODO: fix this for Ahmad
                import appdirs
                cache_dir = Path(appdirs.user_cache_dir(appname="particle-tracking-manager", appauthor="axiom-data-science"))
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.interpolator_filename = cache_dir / Path(f"{self.ocean_model}_interpolator").with_suffix(".pickle")
            else:
                self.interpolator_filename = Path(self.interpolator_filename).with_suffix(".pickle")
            self.save_interpolator = True
            
            # change interpolator_filename to string
            self.interpolator_filename = str(self.interpolator_filename)
            
            if Path(self.interpolator_filename).exists():
                logger.info(f"Loading the interpolator from {self.interpolator_filename}.")
            else:
                logger.info(f"A new interpolator will be saved to {self.interpolator_filename}.")
        else:
            self.save_interpolator = False
            logger.info("Interpolators will not be saved.")

        return self

    @model_validator(mode='after')
    def check_config_drift_model(self) -> Self:
        """Validators related to the drift_model."""
        if self.drift_model == "Leeway":
            if self.stokes_drift:
                raise ValueError("Stokes drift is not available with the Leeway drift model.")
            if self.do3D:
                raise ValueError("do3D must be False with the Leeway drift model.")
            if self.wind_drift_factor is not None:
                raise ValueError("wind_drift_factor cannot be used with the Leeway drift model. Instead it must be None.")
            if self.wind_drift_depth is not None:
                raise ValueError("wind_drift_depth cannot be used with the Leeway drift model. Instead it must be None.")

        elif self.drift_model == "LarvalFish":
            if not self.vertical_mixing:
                raise ValueError("Vertical mixing must be True with the LarvalFish drift model.")
            if not self.do3D:
                raise ValueError("do3D must be True with the LarvalFish drift model.")
            if self.wind_drift_factor is not None:
                raise ValueError("wind_drift_factor cannot be used with the LarvalFish drift model. Instead it must be None.")
            if self.wind_drift_depth is not None:
                raise ValueError("wind_drift_depth cannot be used with the LarvalFish drift model. Instead it must be None.")
            
        return self    


    # @model_validator(mode='after')
    @computed_field
    def drop_vars(self) -> list[str]:
        """Gather variables to drop based on PTMConfig and OpenDriftConfig."""
        # return self.gather_drop_vars()
    # def gather_drop_vars(self) -> Self:
        """Gather variables to drop based on PTMConfig and OpenDriftConfig."""

        # set drop_vars initial values based on the PTM settings, then add to them for the specific model
        drop_vars = self.model_drop_vars
        # drop_vars = [] #DROP VARS WILL ALREADY EXIST HERE
        # don't need w if not 3D movement
        if not self.do3D:
            drop_vars += ["w"]
            logger.info("Dropping vertical velocity (w) because do3D is False")
        else:
            logger.info("Retaining vertical velocity (w) because do3D is True")

        # don't need winds if stokes drift, wind drift, added wind_uncertainty, and vertical_mixing are off
        # It's possible that winds aren't required for every OpenOil simulation but seems like
        # they would usually be required and the cases are tricky to navigate so also skipping for that case.
        if (
            not self.stokes_drift
            and self.wind_drift_factor == 0
            and self.wind_uncertainty == 0
            and self.drift_model != "OpenOil"
            and not self.vertical_mixing
        ):
            drop_vars += ["Uwind", "Vwind", "Uwind_eastward", "Vwind_northward"]
            logger.info(
                "Dropping wind variables because stokes_drift, wind_drift_factor, wind_uncertainty, and vertical_mixing are all off and drift_model is not 'OpenOil'"
            )
        else:
            logger.info(
                "Retaining wind variables because stokes_drift, wind_drift_factor, wind_uncertainty, or vertical_mixing are on or drift_model is 'OpenOil'"
            )

        # only keep salt and temp for LarvalFish or OpenOil
        if self.drift_model not in ["LarvalFish", "OpenOil"]:
            drop_vars += ["salt", "temp"]
            logger.info(
                "Dropping salt and temp variables because drift_model is not LarvalFish nor OpenOil"
            )
        else:
            logger.info(
                "Retaining salt and temp variables because drift_model is LarvalFish or OpenOil"
            )

        # keep some ice variables for OpenOil (though later see if these are used)
        if self.drift_model != "OpenOil":
            drop_vars += ["aice", "uice_eastward", "vice_northward"]
            logger.info(
                "Dropping ice variables because drift_model is not OpenOil"
            )
        else:
            logger.info(
                "Retaining ice variables because drift_model is OpenOil"
            )

        # if using static masks, drop wetdry masks.
        # if using wetdry masks, drop static masks.
        # TODO: is standard_name_mapping working correctly?
        if self.use_static_masks:
            # TODO: Can the mapping include all possible mappings or does it need to be exact?
            # standard_name_mapping.update({"mask_rho": "land_binary_mask"})
            drop_vars += ["wetdry_mask_rho", "wetdry_mask_u", "wetdry_mask_v"]
            logger.info(
                "Dropping wetdry masks because using static masks instead."
            )
        else:
            # standard_name_mapping.update({"wetdry_mask_rho": "land_binary_mask"})
            drop_vars += ["mask_rho", "mask_u", "mask_v", "mask_psi"]
            logger.info(
                "Dropping mask_rho, mask_u, mask_v, mask_psi because using wetdry masks instead."
            )
        return drop_vars    

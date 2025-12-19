"""Defines classes OpenDriftConfig, LeewayModelConfig, OceanDriftModelConfig, OpenOilModelConfig, LarvalFishModelConfig, HarmfulAlgalBloomModelConfig."""

# Standard library imports
import logging

from os import PathLike
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic.fields import FieldInfo
from typing_extensions import Self

# Third-party imports
from particle_tracking_manager.ocean_model_registry import get_model_end_time

from ...config_ocean_model import register_on_the_fly

# Local imports
from ...config_the_manager import TheManagerConfig
from ...ocean_model_registry import get_model_end_time
from .enums import (
    CoastlineActionEnum,
    DiffusivityModelEnum,
    DriftModelEnum,
    DropletSizeDistributionEnum,
    HABSpeciesTypeEnum,
    ObjectTypeEnum,
    OilTypeEnum,
    PlotTypeEnum,
    RadiusTypeEnum,
    SeafloorActionEnum,
)
from .enums.species_types import (
    HAB_SPECIES_LABELS,
    SPECIES_HAB_DEFAULTS,
    SPECIES_HAB_MANAGER_DEFAULTS,
    HABParameters,
    _species_descriptions,
)


logger = logging.getLogger()

# class OpenDriftConfig(BaseModel):
class OpenDriftConfig(TheManagerConfig):
    """Some of the parameters in this mirror OpenDriftSimulation clss in OpenDrift"""

    drift_model: DriftModelEnum = Field(
        default=DriftModelEnum.OceanDrift,  # .value,
        description="Scenario to use for simulation.",
    )

    save_interpolator: bool = Field(
        default=False, description="Whether to save the interpolator."
    )

    interpolator_filename: PathLike[str] | None = Field(
        None,
        description="Filename to save interpolator to or read interpolator from. Exclude suffix (which should be .pickle).",
        json_schema_extra=dict(ptm_level=3),
    )

    export_variables: list[str] | None = Field(
        default=None,
        description="List of variables to export. Options available with `m.all_export_variables` for a given `drift_model`. "
        "['lon', 'lat', 'ID', 'status', 'z'] will always be exported. Default of None means all possible variables are exported.",
        json_schema_extra=dict(ptm_level=3),
    )

    plots: dict[str, dict] | str | None = Field(
        default=None,
        json_schema_extra=dict(ptm_level=1),
        description="Dictionary of plots to generate using OpenDrift.",
    )

    radius: float = Field(
        default=1000.0,
        ge=0.0,
        le=1000000,
        description="Radius around each lon-lat pair, within which particles will be seeded according to `radius_type`.",
        json_schema_extra=dict(
            ptm_level=2,
            units="m",
        ),
    )

    radius_type: RadiusTypeEnum = Field(
        default=RadiusTypeEnum.gaussian,  # .value,
        description="Distribution for seeding particles around location. Options: 'gaussian' or 'uniform'.",
        json_schema_extra=dict(
            ptm_level=3,
        ),
    )

    # OpenDriftSimulation parameters

    max_speed: float = Field(
        default=7.0,
        description="Typical maximum speed of elements, used to estimate reader buffer size",
        gt=0,
        title="Maximum speed",
        json_schema_extra={
            "units": "m/s",
            "od_mapping": "drift:max_speed",
            "ptm_level": 1,
        },
    )

    use_auto_landmask: bool = Field(
        default=False,
        description="If True, use a global-scale land mask from https://www.generic-mapping-tools.org/remote-datasets/earth-mask.html. Dataset scale selected is `auto`. If False, use the land mask from the ocean model.",
        title="Use Auto Landmask",
        json_schema_extra={"od_mapping": "general:use_auto_landmask", "ptm_level": 3},
    )

    coastline_action: CoastlineActionEnum = Field(
        default=CoastlineActionEnum.stranding,  # .value,
        description="This controls particle behavior at the coastline. Use `previous` for a particle to move back to its previous location if it hits land. Use `stranding` to have a particle stick (that is, become deactivated) where it interacts with land. With None, objects may also move over land.",
        title="Coastline Action",
        json_schema_extra={"od_mapping": "general:coastline_action", "ptm_level": 2},
    )

    current_uncertainty: float = Field(
        default=0,
        description="Add gaussian perturbation with this standard deviation to current components at each time step",
        title="Current Uncertainty",
        ge=0,
        le=5,
        json_schema_extra={
            "units": "m/s",
            "od_mapping": "drift:current_uncertainty",
            "ptm_level": 2,
        },
    )

    wind_uncertainty: float = Field(
        default=0,
        description="Add gaussian perturbation with this standard deviation to wind components at each time step.",
        title="Wind Uncertainty",
        ge=0,
        le=5,
        json_schema_extra={
            "units": "m/s",
            "od_mapping": "drift:wind_uncertainty",
            "ptm_level": 2,
        },
    )

    # add od_mapping to what should otherwise be in TheManagerConfig
    horizontal_diffusivity: float | None = FieldInfo.merge_field_infos(
        TheManagerConfig.model_fields["horizontal_diffusivity"],
        Field(json_schema_extra=dict(od_mapping="drift:horizontal_diffusivity")),
    )
    stokes_drift: bool = FieldInfo.merge_field_infos(
        TheManagerConfig.model_fields["stokes_drift"],
        Field(json_schema_extra=dict(od_mapping="drift:stokes_drift")),
    )
    z: float | None = FieldInfo.merge_field_infos(
        TheManagerConfig.model_fields["z"],
        Field(json_schema_extra=dict(od_mapping="seed:z")),
    )
    number: int = FieldInfo.merge_field_infos(
        TheManagerConfig.model_fields["number"],
        Field(json_schema_extra=dict(od_mapping="seed:number")),
    )

    model_config = {
        "validate_defaults": True,
        "use_enum_values": True,
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def check_interpolator_filename(self) -> Self:
        """Check if interpolator_filename is set correctly."""
        if self.interpolator_filename is not None and not self.use_cache:
            raise ValueError(
                "If interpolator_filename is input, use_cache must be True."
            )
        return self

    @model_validator(mode="after")
    def check_config_z_value(self) -> Self:
        """Check if z is set correctly."""
        if hasattr(self, "seed_seafloor"):
            if not self.seed_seafloor and self.z is None:
                raise ValueError("z needs a non-None value if seed_seafloor is False.")
            if self.seed_seafloor and self.z is not None:
                raise ValueError("z needs to be None if seed_seafloor is True.")
        return self

    # this is not true! For example, OpenOil has by default no vertical advection but yes vertical mixing
    # @model_validator(mode="after")
    # def check_config_do3D(self) -> Self:
    #     """Check if do3D is set correctly."""
    #     if hasattr(self, "vertical_mixing"):
    #         if not self.do3D and self.vertical_mixing:
    #             raise ValueError(
    #                 "If do3D is False, vertical_mixing must also be False."
    #             )
    #     return self

    @model_validator(mode="after")
    def setup_interpolator(self) -> Self:
        """Setup interpolator."""

        if self.use_cache:
            if self.interpolator_filename is None:
                import appdirs

                cache_dir = Path(
                    appdirs.user_cache_dir(
                        appname="particle-tracking-manager",
                        appauthor="axiom-data-science",
                    )
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.interpolator_filename = cache_dir / Path(
                    f"{self.ocean_model}_interpolator"
                ).with_suffix(".pickle")
            else:
                self.interpolator_filename = Path(
                    self.interpolator_filename
                ).with_suffix(".pickle")
            self.save_interpolator = True

            # # change interpolator_filename to string
            # self.interpolator_filename = str(self.interpolator_filename)

            logger.debug(f"Interpolator filename: {self.interpolator_filename}")

            if Path(self.interpolator_filename).exists():
                logger.debug(
                    f"Will load the interpolator from {self.interpolator_filename}."
                )
            else:
                logger.debug(
                    f"A new interpolator will be saved to {self.interpolator_filename}."
                )

        else:
            self.save_interpolator = False
            logger.debug("Interpolator will not be saved.")

        return self

    @property
    def drop_vars(self) -> list[str]:
        """Gather variables to drop based on PTMConfig and OpenDriftConfig."""

        # set drop_vars initial values based on the PTM settings, then add to them for the specific model
        drop_vars = (
            self.ocean_model_config.model_drop_vars.copy()
        )  # without copy this remembers drop_vars from other instances

        # don't need w if not 3D movement
        if not self.do3D:
            drop_vars += ["w"]
            logger.debug("Dropping vertical velocity (w) because do3D is False")
        else:
            logger.debug("Retaining vertical velocity (w) because do3D is True")

        # don't need winds if stokes drift, wind drift, added wind_uncertainty, and vertical_mixing are off
        # It's possible that winds aren't required for every OpenOil simulation but seems like
        # they would usually be required and the cases are tricky to navigate so also skipping for that case.
        if (
            not self.stokes_drift
            and (hasattr(self, "wind_drift_factor") and self.wind_drift_factor == 0)
            and self.wind_uncertainty == 0
            and self.drift_model != "OpenOil"
            and not self.vertical_mixing
        ):
            drop_vars += ["Uwind", "Vwind", "Uwind_eastward", "Vwind_northward"]
            logger.debug(
                "Dropping wind variables because stokes_drift, wind_drift_factor, wind_uncertainty, and vertical_mixing are all off and drift_model is not 'OpenOil'"
            )
        else:
            logger.debug(
                "Retaining wind variables because stokes_drift, wind_drift_factor, wind_uncertainty, or vertical_mixing are on or drift_model is 'OpenOil'"
            )

        # only keep salt and temp for LarvalFish or OpenOil
        if self.drift_model not in ["LarvalFish", "OpenOil", "HarmfulAlgalBloom"]:
            drop_vars += ["salt", "temp"]
            logger.debug(
                "Dropping salt and temp variables because drift_model is not LarvalFish nor OpenOil nor HarmfulAlgalBloom"
            )
        else:
            logger.debug(
                "Retaining salt and temp variables because drift_model is LarvalFish or OpenOil or HarmfulAlgalBloom"
            )

        # keep some ice variables for OpenOil (though later see if these are used)
        if self.drift_model != "OpenOil":
            drop_vars += ["aice", "uice_eastward", "vice_northward"]
            logger.debug("Dropping ice variables because drift_model is not OpenOil")
        else:
            logger.debug("Retaining ice variables because drift_model is OpenOil")

        # if using static masks, drop wetdry masks.
        # if using wetdry masks, drop static masks.
        # TODO: is standard_name_mapping working correctly?
        if self.use_static_masks:
            # TODO: Can the mapping include all possible mappings or does it need to be exact?
            # standard_name_mapping.update({"mask_rho": "land_binary_mask"})
            drop_vars += ["wetdry_mask_rho", "wetdry_mask_u", "wetdry_mask_v"]
            logger.debug("Dropping wetdry masks because using static masks instead.")
        else:
            # standard_name_mapping.update({"wetdry_mask_rho": "land_binary_mask"})
            drop_vars += ["mask_rho", "mask_u", "mask_v", "mask_psi"]
            logger.debug(
                "Dropping mask_rho, mask_u, mask_v, mask_psi because using wetdry masks instead."
            )
        return drop_vars

    @model_validator(mode="after")
    def check_plot_oil(self) -> Self:
        """Check if oil budget plot is requested and drift model is OpenOil."""
        if (
            self.plots is not None
            and isinstance(self.plots, dict)
            and "oil" in self.plots.keys()
        ):
            if self.drift_model != "OpenOil":
                raise ValueError(
                    "Oil budget plot only available for OpenOil drift model"
                )
        return self

    @model_validator(mode="after")
    def check_plot_all(self) -> Self:
        """Check if all plots are requested."""
        if (
            self.plots is not None
            and isinstance(self.plots, dict)
            and "all" in self.plots.keys()
            and len(self.plots) > 1
        ):
            raise ValueError(
                "If 'all' is specified for plots, it must be the only plot option."
            )
        return self

    @model_validator(mode="after")
    def check_plot_prefix_enum(self) -> Self:
        """Check if plot keys start with a PlotTypeEnum."""
        if self.plots is not None:
            assert isinstance(self.plots, dict)
            present_keys = [
                key
                for key in self.plots.keys()
                for PlotType in PlotTypeEnum
                if key.startswith(PlotType.value)
            ]
            random_keys = set(self.plots.keys()) - set(present_keys)
            if len(random_keys) > 0:
                raise ValueError(
                    f"Plot keys must start with a PlotTypeEnum. The following keys do not: {random_keys}"
                )
        return self


class LeewayModelConfig(OpenDriftConfig):
    """Leeway model configuration for OpenDrift."""

    drift_model: DriftModelEnum = DriftModelEnum.Leeway

    object_type: ObjectTypeEnum = Field(
        default=ObjectTypeEnum("Person-in-water (PIW), unknown state (mean values)"),
        description="Leeway object category for this simulation. List is originally from USCG technical reports. More details here: https://github.com/OpenDrift/opendrift/blob/master/opendrift/models/OBJECTPROP.DAT.",
        title="Object Type",
        json_schema_extra={"od_mapping": "seed:object_type", "ptm_level": 1},
    )

    # modify default values
    stokes_drift: bool = FieldInfo.merge_field_infos(
        OpenDriftConfig.model_fields["stokes_drift"], Field(default=False)
    )

    @model_validator(mode="after")
    def check_stokes_drift(self) -> Self:
        """Check if stokes_drift is set to False for Leeway model."""
        if self.stokes_drift:
            raise ValueError("stokes_drift must be False with the Leeway drift model.")

        return self

    @model_validator(mode="after")
    def check_do3D(self) -> Self:
        """Check if do3D is set to False for Leeway model."""
        if self.do3D:
            raise ValueError("do3D must be False with the Leeway drift model.")

        return self


class OceanDriftModelConfig(OpenDriftConfig):
    """Ocean drift model configuration for OpenDrift."""

    drift_model: DriftModelEnum = DriftModelEnum.OceanDrift  # .value

    seed_seafloor: bool = Field(
        default=False,
        description="Elements are seeded at seafloor, and seeding depth (z) is neglected and must be None.",
        title="Seed Seafloor",
        json_schema_extra={"od_mapping": "seed:seafloor", "ptm_level": 2},
    )

    diffusivitymodel: DiffusivityModelEnum = Field(
        default="windspeed_Large1994",
        description="Algorithm/source used for profile of vertical diffusivity. Environment means that diffusivity is acquired from readers or environment constants/fallback. Parameterizations based on wind speed are also available.",
        title="Diffusivity model",
        json_schema_extra={
            "units": "seconds",
            "od_mapping": "vertical_mixing:diffusivitymodel",
            "ptm_level": 3,
        },
    )

    mixed_layer_depth: float = Field(
        default=10,
        description="mixed_layer_depth controls how deep the vertical diffusivity profile reaches. This sets the fallback value for ocean_mixed_layer_thickness if not available from any reader.",
        title="Mixed Layer Depth",
        ge=0.0,
        json_schema_extra={
            "units": "m",
            "od_mapping": "environment:mixed_layer_depth",
            "ptm_level": 3,
        },
    )

    seafloor_action: SeafloorActionEnum = Field(
        default=SeafloorActionEnum.lift_to_seafloor,  # .value,
        description="This controls particle behavior at the seafloor. Use `deactivate` to stick particles to the seafloor at the point of interaction. Use `lift_to_seafloor` to elevate particles up to seabed if below. User `previous` to move elements back to previous position. Use None to ignore seafloor.",
        title="Seafloor Action",
        json_schema_extra={
            "od_mapping": "general:seafloor_action",
            "ptm_level": 2,
        },
    )

    wind_drift_depth: float | None = Field(
        default=0.1,
        description="The direct wind drift (windage) is linearly decreasing from the surface value (wind_drift_factor) until 0 at this depth.",
        title="Wind Drift Depth",
        ge=0,
        le=10,
        json_schema_extra={
            "units": "meters",
            "od_mapping": "drift:wind_drift_depth",
            "ptm_level": 3,
        },
    )

    vertical_mixing_timestep: float = Field(
        default=60,
        description="Time step used for inner (fast) loop of the vertical mixing model. Set this smaller to increase frequency of vertical mixing calculation; number of loops is calculated as int(self.time_step/vertical_mixing_timestep) so vertical_mixing_timestep must be smaller than time_step.",
        title="Vertical Mixing Timestep",
        ge=0.1,
        le=3600,
        json_schema_extra={
            "units": "seconds",
            "od_mapping": "vertical_mixing:timestep",
            "ptm_level": 3,
        },
    )

    wind_drift_factor: float = Field(
        default=0.02,
        description="Elements at surface are moved with this fraction of the wind vector, in addition to currents and Stokes drift.",
        title="Wind Drift Factor",
        ge=0,
        json_schema_extra={
            "units": "1",
            "od_mapping": "seed:wind_drift_factor",
            "ptm_level": 2,
        },
    )

    vertical_mixing: bool = Field(
        default=False,
        description="Activate vertical mixing scheme. Vertical mixing includes movement due to buoyancy and turbulent mixing.",
        title="Vertical Mixing",
        json_schema_extra={
            "od_mapping": "drift:vertical_mixing",
            "ptm_level": 2,
        },
    )


class OpenOilModelConfig(OceanDriftModelConfig):
    """OpenOil model configuration for OpenDrift."""

    drift_model: DriftModelEnum = DriftModelEnum.OpenOil  # .value

    oil_type: OilTypeEnum = Field(
        default=OilTypeEnum.AD04012,  # .value,
        description="Oil type to be used for the simulation, from the NOAA ADIOS database.",
        title="Oil Type",
        json_schema_extra={
            "od_mapping": "seed:oil_type",
            "ptm_level": 1,
            "oneOf": [{"const": oil.value, "title": oil.title} for oil in OilTypeEnum],
        },
    )

    m3_per_hour: float = Field(
        default=1,
        description="The amount (volume) of oil released per hour (or total amount if release is instantaneous).",
        title="M3 Per Hour",
        gt=0,
        json_schema_extra={
            "units": "m3 per hour",
            "od_mapping": "seed:m3_per_hour",
            "ptm_level": 1,
        },
    )

    oil_film_thickness: float = Field(
        default=0.001,
        description="Seeding value of oil_film_thickness. Values are calculated by OpenDrift starting from this initial value if `update_oilfilm_thickness==True`.",
        title="Oil Film Thickness",
        json_schema_extra={
            "units": "m",
            "od_mapping": "seed:oil_film_thickness",
            "ptm_level": 3,
        },
    )

    droplet_size_distribution: DropletSizeDistributionEnum = Field(
        default=DropletSizeDistributionEnum.uniform,  # .value,
        description="Droplet size distribution used for subsea release.",
        title="Droplet Size Distribution",
        json_schema_extra={
            "od_mapping": "seed:droplet_size_distribution",
            "ptm_level": 3,
        },
    )

    droplet_diameter_mu: float = Field(
        default=0.001,
        description="The mean diameter of oil droplet for a subsea release, used in normal/lognormal distributions.",
        title="Droplet Diameter Mu",
        ge=1e-08,
        le=1,
        json_schema_extra={
            "units": "meters",
            "od_mapping": "seed:droplet_diameter_mu",
            "ptm_level": 3,
        },
    )

    droplet_diameter_sigma: float = Field(
        default=0.0005,
        description="The standard deviation in diameter of oil droplet for a subsea release, used in normal/lognormal distributions.",
        title="Droplet Diameter Sigma",
        ge=1e-08,
        le=1,
        json_schema_extra={
            "units": "meters",
            "od_mapping": "seed:droplet_diameter_sigma",
            "ptm_level": 3,
        },
    )

    droplet_diameter_min_subsea: float = Field(
        default=0.0005,
        description="The minimum diameter of oil droplet for a subsea release, used in uniform distribution.",
        title="Droplet Diameter Min Subsea",
        ge=1e-08,
        le=1,
        json_schema_extra={
            "units": "meters",
            "od_mapping": "seed:droplet_diameter_min_subsea",
            "ptm_level": 3,
        },
    )

    droplet_diameter_max_subsea: float = Field(
        default=0.005,
        description="The maximum diameter of oil droplet for a subsea release, used in uniform distribution.",
        title="Droplet Diameter Max Subsea",
        ge=1e-08,
        le=1,
        json_schema_extra={
            "units": "meters",
            "od_mapping": "seed:droplet_diameter_max_subsea",
            "ptm_level": 3,
        },
    )

    emulsification: bool = Field(
        default=True,
        description="If True, surface oil is emulsified, i.e. water droplets are mixed into oil due to wave mixing, with resulting increase of viscosity.",
        title="Emulsification",
        json_schema_extra={
            "od_mapping": "processes:emulsification",
            "ptm_level": 2,
        },
    )

    dispersion: bool = Field(
        default=True,
        description="If True, oil is removed from simulation (dispersed), if entrained as very small droplets.",
        title="Dispersion",
        json_schema_extra={
            "od_mapping": "processes:dispersion",
            "ptm_level": 2,
        },
    )

    evaporation: bool = Field(
        default=True,
        description="If True, surface oil is evaporated.",
        title="Evaporation",
        json_schema_extra={
            "od_mapping": "processes:evaporation",
            "ptm_level": 2,
        },
    )

    update_oilfilm_thickness: bool = Field(
        default=True,
        description="If True, Oil film thickness is calculated at each time step. If False, oil film thickness is kept constant with value provided at seeding.",
        title="Update Oilfilm Thickness",
        json_schema_extra={
            "od_mapping": "processes:update_oilfilm_thickness",
            "ptm_level": 2,
        },
    )

    biodegradation: bool = Field(
        default=True,
        description="If True, oil mass is biodegraded (eaten by bacteria).",
        title="Biodegradation",
        json_schema_extra={
            "od_mapping": "processes:biodegradation",
            "ptm_level": 2,
        },
    )

    # overwrite the defaults from OceanDriftModelConfig for a few inherited parameters,
    # but don't want to have to repeat the full definition
    current_uncertainty: float = FieldInfo.merge_field_infos(
        OceanDriftModelConfig.model_fields["current_uncertainty"], Field(default=0.0)
    )
    wind_uncertainty: float = FieldInfo.merge_field_infos(
        OceanDriftModelConfig.model_fields["wind_uncertainty"], Field(default=0.0)
    )
    wind_drift_factor: float = FieldInfo.merge_field_infos(
        OceanDriftModelConfig.model_fields["wind_drift_factor"], Field(default=0.03)
    )
    vertical_mixing: bool = FieldInfo.merge_field_infos(
        OceanDriftModelConfig.model_fields["vertical_mixing"], Field(default=True)
    )


class LarvalFishModelConfig(OceanDriftModelConfig):
    """Larval fish model configuration for OpenDrift."""

    drift_model: DriftModelEnum = DriftModelEnum.LarvalFish  # .value

    diameter: float = Field(
        default=0.0014,
        description="Seeding value of diameter. The diameter gives the egg diameter so must be used with `hatched=0`.",
        title="Diameter",
        gt=0,
        json_schema_extra={
            "units": "m",
            "od_mapping": "seed:diameter",
            "ptm_level": 2,
        },
    )

    neutral_buoyancy_salinity: float = Field(
        default=31.25,
        description="Seeding value of neutral_buoyancy_salinity. This is a property of the egg so must be used with `hatched=0`.",
        title="Neutral Buoyancy Salinity",
        gt=0,
        json_schema_extra={
            "units": "PSU",
            "od_mapping": "seed:neutral_buoyancy_salinity",
            "ptm_level": 2,
        },
    )

    stage_fraction: float | None = Field(
        default=0.0,
        description="Seeding value of stage_fraction. stage_fraction tracks percentage of development time completed, from 0 to 1, where a value of 1 means the egg has hatched. If `hatched==1` then `stage_fraction` is ignored in `OpenDrift`, but has to be None.",
        title="Stage Fraction",
        ge=0,
        le=1,
        json_schema_extra={
            "units": "",
            "od_mapping": "seed:stage_fraction",
            "ptm_level": 2,
        },
    )

    hatched: int = Field(
        default=0,
        description="Seeding value of hatched. 0 for eggs, 1 for larvae.",
        title="Hatched",
        ge=0,
        le=1,
        json_schema_extra={
            "units": "",
            "od_mapping": "seed:hatched",
            "ptm_level": 2,
        },
    )

    length: float | None = Field(
        default=None,
        description="Seeding value of length. This is not currently used.",
        title="Length",
        gt=0,
        json_schema_extra={
            "units": "mm",
            "od_mapping": "seed:length",
            "ptm_level": 2,
        },
    )

    weight: float = Field(
        default=0.08,
        description="Seeding value of weight. This is the starting weight for larval fish, whenever they reach that stage.",
        title="Weight",
        gt=0,
        json_schema_extra={
            "units": "mg",
            "od_mapping": "seed:weight",
            "ptm_level": 2,
        },
    )

    # override inherited parameter defaults
    vertical_mixing: bool = FieldInfo.merge_field_infos(
        OceanDriftModelConfig.model_fields["vertical_mixing"], Field(default=True)
    )
    do3D: bool = FieldInfo.merge_field_infos(
        TheManagerConfig.model_fields["do3D"], Field(default=True)
    )

    @model_validator(mode="after")
    def check_do3D(self) -> Self:
        """Check if do3D is set to True for LarvalFish model."""
        if not self.do3D:
            raise ValueError("do3D must be True with the LarvalFish drift model.")

        return self

    @model_validator(mode="after")
    def check_vertical_mixing(self) -> Self:
        """Check if vertical_mixing is set to True for LarvalFish model."""
        if not self.vertical_mixing:
            raise ValueError(
                "vertical_mixing must be True with the LarvalFish drift model."
            )

        return self

    @model_validator(mode="after")
    def check_hatched_and_stage_fraction(self) -> Self:
        """If hatched==1, stage_fraction should be None.

        This only applies for seeding, not for the simulation.
        """

        if self.hatched == 1 and self.stage_fraction is not None:
            raise ValueError("If hatched==1, stage_fraction should be None.")
        return self


class HarmfulAlgalBloomModelConfig(HABParameters, OceanDriftModelConfig):
    """Harmful algal bloom model configuration for OpenDrift."""

    drift_model: DriftModelEnum = DriftModelEnum.HarmfulAlgalBloom
    # import pdb; pdb.set_trace()
    species_type: HABSpeciesTypeEnum = Field(
        default=HABSpeciesTypeEnum.PN,
        description="HarmfulAlgalBloom species category for this simulation. This option maps to individual properties which can instead be set manually if desired.",
        title="HAB Species Type",
        json_schema_extra={
            "ptm_level": 1,
            "oneOf": [
                {
                    "const": species.value,
                    "title": HAB_SPECIES_LABELS[species],
                    "description": _species_descriptions[species.value],
                }
                for species in HABSpeciesTypeEnum
            ],
        },
    )

    # # new field: full parameter bundle for the HAB species
    # hab_params: HABParameters = Field(
    #     description=(
    #         "Biological parameters for the HAB model. If omitted or partially "
    #         "specified, defaults are taken from the selected species_type and "
    #         "overridden by any provided values."
    #     ),
    #     json_schema_extra={"ptm_level": 1},
    # )

    # temperature_death_min: float = Field(
    #     # default=3.0,
    #     description="Minimum temperature for living. Below this temperature, cells have high mortality rate. Between this and temperature_viable_min, cells have no growth.",
    #     title="Boundary temperature for cell death.",
    #     # gt=-3,
    #     json_schema_extra={
    #         "units": "degrees",
    #         "od_mapping": "hab:temperature_death_min",
    #         "ptm_level": 2,
    #     },
    # )

    # # override inherited parameter defaults
    # vertical_mixing: bool = FieldInfo.merge_field_infos(
    #     OceanDriftModelConfig.model_fields["vertical_mixing"], Field(default=True)
    # )
    # do3D: bool = FieldInfo.merge_field_infos(
    #     TheManagerConfig.model_fields["do3D"], Field(default=True)
    # )

    @model_validator(mode="before")
    @classmethod
    def apply_species_defaults(cls, data: Any) -> Any:
        """
        - If species_type has presets:
            * Start from species HAB defaults
            * Overlay any user-provided hab_params
            * Set z/do3D defaults only if user didn't supply them
        - If species_type has no presets (e.g., Custom):
            * Require user to provide hab_params explicitly
            * z/do3D behave as usual (user or base defaults)
        """
        if not isinstance(data, dict):
            return data

        # Ensure species_type has some value in raw input
        data.setdefault("species_type", HABSpeciesTypeEnum.PN)
        species = data["species_type"]

        # -------- HAB param defaults (flattened) --------
        if species in SPECIES_HAB_DEFAULTS:
            # species defaults
            default_params = SPECIES_HAB_DEFAULTS[species].model_dump()

            # apply defaults only where user did not provide a value
            for field_name, default_value in default_params.items():
                data.setdefault(field_name, default_value)
        else:
            # species has no preset (e.g. Custom): require all HAB fields
            required_fields = HABParameters.model_fields.keys()
            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValueError(
                    "Custom HAB species requires explicit values for all HAB parameters. "
                    f"Missing: {', '.join(missing)}"
                )

        # -------- Other config field defaults (e.g. z, do3D) --------
        field_defaults = SPECIES_HAB_MANAGER_DEFAULTS.get(species, {})
        for field_name, default_value in field_defaults.items():
            # only apply if the user did not supply this field
            data.setdefault(field_name, default_value)

        return data

    # @model_validator(mode="after")
    # def setup_species_parameters(self) -> Self:
    #     """Assign species-specific parameters."""

    #     if self.species_type == HABSpeciesTypeEnum("Pseudo_nitzschia"):

    #         if self.do3D:
    #             raise ValueError("Pseudo_nitzschia species requires do3D to be False.")
    #         if self.z != 0.0:
    #             raise ValueError("Pseudo_nitzschia species requires z to be 0.0 m.")
    #         logger.debug("HAB species Pseudo_nitzschia selected.")

    #     else:
    #         raise ValueError(
    #             f"Species type {self.species_type} not recognized for HarmfulAlgalBloom model."
    #         )
    #     return self

    # @model_validator(mode="after")
    # def check_do3D(self) -> Self:
    #     """Check if do3D is set to True for LarvalFish model."""
    #     if not self.do3D:
    #         raise ValueError("do3D must be True with the LarvalFish drift model.")

    #     return self

    # @model_validator(mode="after")
    # def check_vertical_mixing(self) -> Self:
    #     """Check if vertical_mixing is set to True for LarvalFish model."""
    #     if not self.vertical_mixing:
    #         raise ValueError(
    #             "vertical_mixing must be True with the LarvalFish drift model."
    #         )

    #     return self

    # @model_validator(mode="after")
    # def check_hatched_and_stage_fraction(self) -> Self:
    #     """If hatched==1, stage_fraction should be None.

    #     This only applies for seeding, not for the simulation.
    #     """

    #     if self.hatched == 1 and self.stage_fraction is not None:
    #         raise ValueError("If hatched==1, stage_fraction should be None.")
    #     return self


open_drift_mapper: dict[str, type[OpenDriftConfig]] = {
    "OceanDrift": OceanDriftModelConfig,
    "OpenOil": OpenOilModelConfig,
    "LarvalFish": LarvalFishModelConfig,
    "Leeway": LeewayModelConfig,
    "HarmfulAlgalBloom": HarmfulAlgalBloomModelConfig,
}

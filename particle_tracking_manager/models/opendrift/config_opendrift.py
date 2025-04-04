"""Defines classes OpenDriftConfig, LeewayModelConfig, OceanDriftModelConfig, OpenOilModelConfig, and LarvalFishModelConfig."""

# Standard library imports
import logging
from enum import Enum
from pathlib import Path

# Third-party imports
from pydantic import Field, model_validator
from pydantic.fields import FieldInfo
from typing_extensions import Self

# Local imports
from ...config_the_manager import TheManagerConfig


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
    

# Define Pydantic Enum classes
class DiffusivityModelEnum(str, Enum):
    environment = "environment"
    stepfunction = "stepfunction"
    windspeed_Sundby1983 = "windspeed_Sundby1983"
    windspeed_Large1994 = "windspeed_Large1994"
    gls_tke = "gls_tke"
    constant = "constant"

class CoastlineActionEnum(str, Enum):
    none = "none"
    stranding = "stranding"
    previous = "previous"

class SeafloorActionEnum(str, Enum):
    none = "none"
    lift_to_seafloor = "lift_to_seafloor"
    deactivate = "deactivate"
    previous = "previous"

class PlotTypeEnum(str, Enum):
    spaghetti = "spaghetti"
    animation = "animation"
    animation_profile = "animation_profile"
    oil = "oil"
    property = "property"
    all = "all"


# class OpenDriftConfig(BaseModel):
class OpenDriftConfig(TheManagerConfig):
    """Some of the parameters in this mirror OpenDriftSimulation clss in OpenDrift"""
    drift_model: DriftModelEnum = Field(default=DriftModelEnum.OceanDrift.value, description="Drift model to use for simulation.")
    
    save_interpolator: bool = Field(default=False, description="Whether to save the interpolator.")

    interpolator_filename: str | None = Field(None, description="Filename to save interpolator to or read interpolator from. Exclude suffix (which should be .pickle).", json_schema_extra=dict(ptm_level=3))
    
    export_variables: list[str] | None = Field(
        default=None,
        description="List of variables to export. Options available with `m.all_export_variables` for a given `drift_model`. "
                    "['lon', 'lat', 'ID', 'status', 'z'] will always be exported. Default of None means all possible variables are exported.",
        json_schema_extra=dict(ptm_level=3),
    )
    
    plots: dict[str, dict] | None = Field(default=None, json_schema_extra=dict(ptm_level=1), description="Dictionary of plots to generate using OpenDrift.")
    
    radius: float = Field(
        default=1000.0, 
        ge=0.0, le=1000000, description="Radius around each lon-lat pair, within which particles will be randomly seeded.", json_schema_extra=dict(ptm_level=2, units="m", ))
    
    radius_type: RadiusTypeEnum = Field(
        default=RadiusTypeEnum.gaussian.value, 
        description="Radius type. Options: 'gaussian' or 'uniform'.", json_schema_extra=dict(ptm_level=3, ))
    
    # OpenDriftSimulation parameters

    max_speed: float = Field(
        default=5.0,
        description="Typical maximum speed of elements, used to estimate reader buffer size",
        gt=0,
        title="Maximum speed",
        json_schema_extra={"units": "m/s", "od_mapping": "drift:max_speed", "ptm_level": 1},
    )

    
    use_auto_landmask: bool = Field(
        default=True,
        description="A built-in GSHHG global landmask is used if True, otherwise landmask is taken from reader or fallback value.",
        title="Use Auto Landmask",
        json_schema_extra={"od_mapping": "general:use_auto_landmask", "ptm_level": 3},
    )

    coastline_action: CoastlineActionEnum = Field(
        default=CoastlineActionEnum.stranding.value,
        description="None means that objects may also move over land. stranding means that objects are deactivated if they hit land. previous means that objects will move back to the previous location if they hit land",
        title="Coastline Action",
        json_schema_extra={"od_mapping": "general:coastline_action", "ptm_level": 2},
    )

    current_uncertainty: float = Field(
        default=0,
        description="Add gaussian perturbation with this standard deviation to current components at each time step",
        title="Current Uncertainty",
        ge=0,
        le=5,
        json_schema_extra={"units": "m/s", "od_mapping": "drift:current_uncertainty", "ptm_level": 2},
    )
    
    wind_uncertainty: float = Field(
        default=0,
        description="Add gaussian perturbation with this standard deviation to wind components at each time step.",
        title="Wind Uncertainty",
        ge=0,
        le=5,
        json_schema_extra={"units": "m/s", "od_mapping": "drift:wind_uncertainty", "ptm_level": 2},
    )

    # add od_mapping to what should otherwise be in TheManagerConfig
    horizontal_diffusivity: float | None = FieldInfo.merge_field_infos(TheManagerConfig.model_fields['horizontal_diffusivity'],
                                                             Field(json_schema_extra=dict(od_mapping="drift:horizontal_diffusivity")))
    stokes_drift: bool = FieldInfo.merge_field_infos(TheManagerConfig.model_fields['stokes_drift'],
                                                             Field(json_schema_extra=dict(od_mapping="drift:stokes_drift")))
    z: float | None = FieldInfo.merge_field_infos(TheManagerConfig.model_fields['z'],
                                                             Field(json_schema_extra=dict(od_mapping="seed:z")))
    number: int = FieldInfo.merge_field_infos(TheManagerConfig.model_fields['number'],
                                                             Field(json_schema_extra=dict(od_mapping="seed:number")))
    time_step: float = FieldInfo.merge_field_infos(TheManagerConfig.model_fields['time_step'],
                                                             Field(json_schema_extra=dict(od_mapping='general:time_step_minutes')))
    time_step_output: float = FieldInfo.merge_field_infos(TheManagerConfig.model_fields['time_step_output'],
                                                             Field(json_schema_extra=dict(od_mapping='general:time_step_output_minutes')))
    

    model_config = {
        "validate_defaults": True,
        "use_enum_values": True,
        "extra": "forbid",
    }

    @model_validator(mode='after')
    def check_interpolator_filename(self) -> Self:
        if self.interpolator_filename is not None and not self.use_cache:
            raise ValueError("If interpolator_filename is input, use_cache must be True.")
        return self


    @model_validator(mode='after')
    def check_config_z_value(self) -> Self:
        if hasattr(self, "seed_seafloor"):
            if not self.seed_seafloor and self.z is None:
                raise ValueError("z needs a non-None value if seed_seafloor is False.")
            if self.seed_seafloor and self.z is not None:
                raise ValueError("z needs to be None if seed_seafloor is True.")
        return self

    @model_validator(mode='after')
    def check_config_do3D(self) -> Self:
        if hasattr(self, "vertical_mixing"):
            if not self.do3D and self.vertical_mixing:
                raise ValueError("If do3D is False, vertical_mixing must also be False.")
        return self
    
    
    @model_validator(mode='after')
    def setup_interpolator(self) -> Self:
        """Setup interpolator."""

        if self.use_cache:
            if self.interpolator_filename is None:
                import appdirs
                cache_dir = Path(appdirs.user_cache_dir(appname="particle-tracking-manager", appauthor="axiom-data-science"))
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.interpolator_filename = cache_dir / Path(f"{self.ocean_model}_interpolator").with_suffix(".pickle")
            else:
                self.interpolator_filename = Path(self.interpolator_filename).with_suffix(".pickle")
            self.save_interpolator = True
            
            # change interpolator_filename to string
            self.interpolator_filename = str(self.interpolator_filename)
            
            logger.debug(f"Interpolator filename: {self.interpolator_filename}")

        else:
            self.save_interpolator = False
            logger.debug("Interpolator will not be saved.")

        return self

    @property
    def drop_vars(self) -> list[str]:
        """Gather variables to drop based on PTMConfig and OpenDriftConfig."""

        # set drop_vars initial values based on the PTM settings, then add to them for the specific model
        drop_vars = self.ocean_model_config.model_drop_vars.copy()  # without copy this remembers drop_vars from other instances

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
        if self.drift_model not in ["LarvalFish", "OpenOil"]:
            drop_vars += ["salt", "temp"]
            logger.debug(
                "Dropping salt and temp variables because drift_model is not LarvalFish nor OpenOil"
            )
        else:
            logger.debug(
                "Retaining salt and temp variables because drift_model is LarvalFish or OpenOil"
            )

        # keep some ice variables for OpenOil (though later see if these are used)
        if self.drift_model != "OpenOil":
            drop_vars += ["aice", "uice_eastward", "vice_northward"]
            logger.debug(
                "Dropping ice variables because drift_model is not OpenOil"
            )
        else:
            logger.debug(
                "Retaining ice variables because drift_model is OpenOil"
            )

        # if using static masks, drop wetdry masks.
        # if using wetdry masks, drop static masks.
        # TODO: is standard_name_mapping working correctly?
        if self.use_static_masks:
            # TODO: Can the mapping include all possible mappings or does it need to be exact?
            # standard_name_mapping.update({"mask_rho": "land_binary_mask"})
            drop_vars += ["wetdry_mask_rho", "wetdry_mask_u", "wetdry_mask_v"]
            logger.debug(
                "Dropping wetdry masks because using static masks instead."
            )
        else:
            # standard_name_mapping.update({"wetdry_mask_rho": "land_binary_mask"})
            drop_vars += ["mask_rho", "mask_u", "mask_v", "mask_psi"]
            logger.debug(
                "Dropping mask_rho, mask_u, mask_v, mask_psi because using wetdry masks instead."
            )
        return drop_vars    

    @model_validator(mode='after')
    def check_plot_oil(self) -> Self:
        if self.plots is not None and "oil" in self.plots.keys():
            if self.drift_model != "OpenOil":
                raise ValueError("Oil budget plot only available for OpenOil drift model")
        return self

    @model_validator(mode='after')
    def check_plot_all(self) -> Self:
        if self.plots is not None and "all" in self.plots.keys() and len(self.plots) > 1:
            raise ValueError("If 'all' is specified for plots, it must be the only plot option.")
        return self

    @model_validator(mode='after')
    def check_plot_prefix_enum(self) -> Self:
        if self.plots is not None:
            present_keys = [key for key in self.plots.keys() for PlotType in PlotTypeEnum if key.startswith(PlotType.value)]
            random_keys = set(self.plots.keys()) - set(present_keys)
            if len(random_keys) > 0:
                raise ValueError(f"Plot keys must start with a PlotTypeEnum. The following keys do not: {random_keys}")
        return self


class ObjectTypeEnum(str, Enum):
    PERSON_IN_WATER_UNKNOWN = "Person-in-water (PIW), unknown state (mean values)"
    PIW_VERTICAL_PFD_TYPE_III_CONSCIOUS = ">PIW, vertical PFD type III conscious"
    PIW_SITTING_PFD_TYPE_I_OR_II = ">PIW, sitting, PFD type I or II"
    PIW_SURVIVAL_SUIT_FACE_UP = ">PIW, survival suit (face up)"
    PIW_SCUBA_SUIT_FACE_UP = ">PIW, scuba suit (face up)"
    PIW_DECEASED_FACE_DOWN = ">PIW, deceased (face down)"
    LIFE_RAFT_DEEP_BALLAST_GENERAL = "Life raft, deep ballast (DB) system, general, unknown capacity and loading (mean values)"
    LIFE_RAFT_4_14_PERSON_CANOPY_AVERAGE = ">4-14 person capacity, deep ballast system, canopy (average)"
    LIFE_RAFT_4_14_PERSON_NO_DROGUE = ">>4-14 person capacity, deep ballast system, no drogue"
    LIFE_RAFT_4_14_PERSON_CANOPY_NO_DROGUE_LIGHT = ">>>4-14 person capacity, deep ballast system, canopy, no drogue, light loading"
    LIFE_RAFT_4_14_PERSON_NO_DROGUE_HEAVY = ">>>4-14 person capacity, deep ballast system, no drogue, heavy loading"
    LIFE_RAFT_4_14_PERSON_CANOPY_WITH_DROGUE_AVERAGE = ">>4-14 person capacity, deep ballast system, canopy, with drogue (average)"
    LIFE_RAFT_4_14_PERSON_CANOPY_WITH_DROGUE_LIGHT = ">>>4-14 person capacity, deep ballast system, canopy, with drogue, light loading"
    LIFE_RAFT_4_14_PERSON_CANOPY_WITH_DROGUE_HEAVY = ">>>4-14 person capacity, deep ballast system, canopy, with drogue, heavy loading"
    LIFE_RAFT_15_50_PERSON_CANOPY_GENERAL = ">15-50 person capacity, deep ballast system, canopy, general (mean values)"
    LIFE_RAFT_15_50_PERSON_CANOPY_NO_DROGUE_LIGHT = ">>15-50 person capacity, deep ballast system, canopy, no drogue, light loading"
    LIFE_RAFT_15_50_PERSON_CANOPY_WITH_DROGUE_HEAVY = ">>15-50 person capacity, deep ballast system, canopy, with drogue, heavy loading"
    DEEP_BALLAST_CAPSIZED = "Deep ballast system, general (mean values), capsized"
    DEEP_BALLAST_SWAMPED = "Deep ballast system, general (mean values), swamped"
    LIFE_RAFT_SHALLOW_BALLAST_CANOPY_GENERAL = "Life-raft, shallow ballast (SB) system AND canopy, general (mean values)"
    LIFE_RAFT_SHALLOW_BALLAST_CANOPY_NO_DROGUE = ">Life-raft, shallow ballast system, canopy, no drogue"
    LIFE_RAFT_SHALLOW_BALLAST_CANOPY_WITH_DROGUE = ">Life-raft, shallow ballast system AND canopy, with drogue"
    LIFE_RAFT_SHALLOW_BALLAST_CANOPY_CAPSIZED = "Life-raft, shallow ballast system AND canopy, capsized"
    LIFE_RAFT_SHALLOW_BALLAST_NAVY_SEIE_NO_DROGUE = "Life Raft - Shallow ballast, canopy, Navy Sub Escape (SEIE) 1-man raft, NO drogue"
    LIFE_RAFT_SHALLOW_BALLAST_NAVY_SEIE_WITH_DROGUE = "Life Raft - Shallow ballast, canopy, Navy Sub Escape (SEIE) 1-man raft, with drogue"
    LIFE_RAFT_NO_BALLAST_GENERAL = "Life-raft, no ballast (NB) system, general (mean values)"
    LIFE_RAFT_NO_BALLAST_NO_CANOPY_NO_DROGUE = ">Life-raft, no ballast system, no canopy, no drogue"
    LIFE_RAFT_NO_BALLAST_NO_CANOPY_WITH_DROGUE = ">Life-raft, no ballast system, no canopy, with drogue"
    LIFE_RAFT_NO_BALLAST_WITH_CANOPY_NO_DROGUE = ">Life-raft, no ballast system, with canopy, no drogue"
    LIFE_RAFT_NO_BALLAST_WITH_CANOPY_WITH_DROGUE = ">Life-raft, no ballast system, with canopy, with drogue"
    SURVIVAL_CRAFT_USCG_SEA_RESCUE_KIT = "Survival Craft - USCG Sea Rescue Kit - 3 ballasted life rafts and 300 meter of line"
    LIFE_RAFT_4_6_PERSON_NO_BALLAST_WITH_CANOPY_NO_DROGUE = "Life-raft, 4-6 person capacity, no ballast, with canopy, no drogue"
    EVACUATION_SLIDE_WITH_LIFE_RAFT = "Evacuation slide with life-raft, 46 person capacity"
    SURVIVAL_CRAFT_SOLAS_HARD_SHELL = "Survival Craft - SOLAS Hard Shell Life Capsule, 22 man"
    # SURVIVAL_CRAFT_OVATEK_HARD_SHELL_LIGHT_NO_DROGUE = "Survival Craft - Ovatek Hard Shell Life Raft, 4 and 7-man, lightly loaded, no drogue (average)"
    SURVIVAL_CRAFT_OVATEK_HARD_SHELL_FULLY_DROGUED = "Survival Craft - Ovatek Hard Shell Life Raft, 4 and 7-man, fully loaded, drogued (average)"
    SURVIVAL_CRAFT_OVATEK_HARD_SHELL_4_MAN_LIGHT_NO_DROGUE = ">Survival Craft - Ovatek Hard Shell Life Raft, 4 man, lightly loaded, no drogue"
    SURVIVAL_CRAFT_OVATEK_HARD_SHELL_7_MAN_LIGHT_NO_DROGUE = ">Survival Craft - Ovatek Hard Shell Life Raft, 7 man, lightly loaded, no drogue"
    SURVIVAL_CRAFT_OVATEK_HARD_SHELL_4_MAN_FULLY_NO_DROGUE = ">Survival Craft - Ovatek Hard Shell Life Raft, 4 man, fully loaded, drogued"
    SURVIVAL_CRAFT_OVATEK_HARD_SHELL_7_MAN_FULLY_NO_DROGUE = ">Survival Craft - Ovatek Hard Shell Life Raft, 7 man, fully loaded, drogued"
    SEA_KAYAK_PERSON_ON_AFT_DECK = "Sea Kayak with person on aft deck"
    SURF_BOARD_PERSON = "Surf board with person"
    WINDSURFER_MAST_AND_SAIL_IN_WATER = "Windsurfer with mast and sail in water"
    SKIFF_MODIFIED_V = "Skiff - modified-v, cathedral-hull, runabout outboard powerboat"
    SKIFF_V_HULL = "Skiff, V-hull"
    SKIFFS_SWAMPED_AND_CAPSIZED = "Skiffs, swamped and capsized"
    SPORT_BOAT_MODIFIED_V_HULL = "Sport boat, no canvas (*1), modified V-hull"
    SPORT_FISHER_CENTER_CONSOLE = "Sport fisher, center console (*2), open cockpit"
    FISHING_VESSEL_GENERAL = "Fishing vessel, general (mean values)"
    FISHING_VESSEL_HAWAIIAN_SAMPAN = "Fishing vessel, Hawaiian Sampan (*3)"
    FISHING_VESSEL_JAPANESE_SIDE_STERN_TRAWLER = ">Fishing vessel, Japanese side-stern trawler"
    FISHING_VESSEL_JAPANESE_LONGLINER = ">Fishing vessel, Japanese Longliner (*3)"
    FISHING_VESSEL_KOREAN = ">Fishing vessel, Korean fishing vessel (*4)"
    FISHING_VESSEL_GILL_NETTER = ">Fishing vessel, Gill-netter with rear reel (*3)"
    COASTAL_FREIGHTER = "Coastal freighter. (*5)"
    SAILBOAT_MONO_HULL = "Sailboat Mono-hull (Average)"
    SAILBOAT_MONO_HULL_DISMASTED = ">Sailboat Mono-hull (Dismasted, Average)"
    SAILBOAT_MONO_HULL_DISMASTED_RUDDER = ">>Sailboat Mono-hull (Dismasted - rudder amidships)"
    SAILBOAT_MONO_HULL_DISMASTED_RUDDER_MISSING = ">>Sailboat Mono-hull (Dismasted - rudder missing)"
    SAILBOAT_MONO_HULL_BARE_MASTED = ">Sailboat Mono-hull (Bare-masted,  Average)"
    SAILBOAT_MONO_HULL_BARE_MASTED_RUDDER = ">>Sailboat Mono-hull (Bare-masted, rudder amidships)"
    SAILBOAT_MONO_HULL_BARE_MASTED_RUDDER_HOVE_TO = ">>Sailboat Mono-hull (Bare-masted, rudder hove-to)"
    SAILBOAT_MONO_HULL_FIN_KEEL = "Sailboat Mono-hull, fin keel, shallow draft (was SAILBOAT-2)"
    SUNFISH_SAILING_DINGY = "Sunfish sailing dingy  -  Bare-masted, rudder missing"
    FISHING_VESSEL_DEBRIS = "Fishing vessel debris"
    SELF_LOCATING_DATUM_MARKER_BUOY = "Self-locating datum marker buoy - no windage"
    NAVY_SUBMARINE_EPIRB = "Navy Submarine EPIRB (SEPIRB)"
    BAIT_WHARF_BOX = "Bait/wharf box, holds a cubic metre of ice, mean values (*6)"
    BAIT_WHARF_BOX_FULL_LOAD = ">Bait/wharf box, holds a cubic metre of ice, full loaded"
    OIL_DRUM = "55-gallon (220 l) Oil Drum"
    CONTAINER_40_FT = "Scaled down (1:3) 40-ft Container (70% submerged)"
    CONTAINER_20_FT = "20-ft Container (80% submerged)"
    WWII_L_MK2_MINE = "WWII L-MK2 mine"
    IMMIGRATION_VESSEL_NO_SAIL = "Immigration vessel, Cuban refugee-raft, no sail (*7)"
    IMMIGRATION_VESSEL_WITH_SAIL = "Immigration vessel, Cuban refugee-raft, with sail (*7)"
    SEWAGE_FLOATABLES = "Sewage floatables, tampon applicator"
    MEDICAL_WASTE = "Medical waste (mean values)"
    MEDICAL_WASTE_VIALS = ">Medical waste, vials"
    MEDICAL_WASTE_VIALS_LARGE = ">>Medical waste, vials, large"
    MEDICAL_WASTE_VIALS_SMALL = ">>Medical waste, vials, small"
    MEDICAL_WASTE_SYRINGES = ">Medical waste, syringes"
    MEDICAL_WASTE_SYRINGES_LARGE = ">>Medical waste, syringes, large"
    MEDICAL_WASTE_SYRINGES_SMALL = ">>Medical waste, syringes, small"

class LeewayModelConfig(OpenDriftConfig):
    drift_model: DriftModelEnum = DriftModelEnum.Leeway.value
    
    object_type: ObjectTypeEnum = Field(
        default=ObjectTypeEnum.PERSON_IN_WATER_UNKNOWN.value,
        description="Leeway object category for this simulation",
        title="Object Type",
        json_schema_extra={"od_mapping": "seed:object_type", "ptm_level": 1},
    )

    # modify default values
    stokes_drift: bool = FieldInfo.merge_field_infos(OpenDriftConfig.model_fields['stokes_drift'],
                                                             Field(default=False))


    @model_validator(mode='after')
    def check_stokes_drift(self) -> Self:
        if self.stokes_drift:
            raise ValueError("stokes_drift must be False with the Leeway drift model.")
            
        return self

    @model_validator(mode='after')
    def check_do3D(self) -> Self:
        if self.do3D:
            raise ValueError("do3D must be False with the Leeway drift model.")
            
        return self


class OceanDriftModelConfig(OpenDriftConfig):
    drift_model: DriftModelEnum = DriftModelEnum.OceanDrift.value
    
    seed_seafloor: bool = Field(
        default=False,
        description="Elements are seeded at seafloor, and seeding depth (z) is neglected.",
        title="Seed Seafloor",
        json_schema_extra={"od_mapping": "seed:seafloor", "ptm_level": 2},
    )
    
    diffusivitymodel: DiffusivityModelEnum = Field(
        default=DiffusivityModelEnum.environment.value,
        description="Algorithm/source used for profile of vertical diffusivity. Environment means that diffusivity is acquired from readers or environment constants/fallback.",
        title="Diffusivity model",
        json_schema_extra={
            "units": "seconds",
            "od_mapping": "vertical_mixing:diffusivitymodel",
            "ptm_level": 3,
        },
    )
    
    mixed_layer_depth: float = Field(
        default=50,
        description="Fallback value for ocean_mixed_layer_thickness if not available from any reader",
        title="Mixed Layer Depth",
        ge=0.0,
        json_schema_extra={
            "units": "m",
            "od_mapping": "environment:mixed_layer_depth",
            "ptm_level": 3,
        },
    )
    
    seafloor_action: SeafloorActionEnum = Field(
        default=SeafloorActionEnum.lift_to_seafloor.value,
        description="deactivate: elements are deactivated; lift_to_seafloor: elements are lifted to seafloor level; previous: elements are moved back to previous position; none; seafloor is ignored.",
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
        description="Time step used for inner loop of vertical mixing.",
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
        description="Elements at surface are moved with this fraction of the wind vector, in addition to currents and Stokes drift",
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
        description="Activate vertical mixing scheme with inner loop",
        title="Vertical Mixing",
        json_schema_extra={
            "od_mapping": "vertical_mixing:vertical_mixing",
            "ptm_level": 2,
        },
    )

# # Make OilTypeEnum with:
# from enum import Enum
# import opendrift.models.openoil.adios.dirjs as dirjs
# oil_strings = [f"{oil.name} ({oil.id})" for oil in dirjs.oils(limit=1300)]
# DynamicEnum = Enum('DynamicEnum', {string.replace(" ","_").replace("(","").replace(")","").replace("U.S.","US").replace(",","").replace("#","").replace("/","").replace("-","_").replace(".","").replace("[","").replace("]","").replace("&","").replace("%","").replace(":",""): string for string in oil_strings})
# DynamicEnum.__members__.values()  # then convert to proper syntax

class OilTypeEnum(str, Enum):
    ABU_SAFAH_ARAMCO_AD00010="ABU SAFAH, ARAMCO (AD00010)"
    ALASKA_NORTH_SLOPE_AD00020="ALASKA NORTH SLOPE (AD00020)"
    ALBERTA_AD00024="ALBERTA (AD00024)"
    ALBERTA_SWEET_MIXED_BLEND_AD00025="ALBERTA SWEET MIXED BLEND (AD00025)"
    ALGERIAN_BLEND_AD00026="ALGERIAN BLEND (AD00026)"
    ALGERIAN_CONDENSATE_CITGO_AD00028="ALGERIAN CONDENSATE, CITGO (AD00028)"
    AMAULIGAK_AD00031="AMAULIGAK (AD00031)"
    ARABIAN_EXTRA_LIGHT_PHILLIPS_AD00039="ARABIAN EXTRA LIGHT, PHILLIPS (AD00039)"
    ARABIAN_EXTRA_LIGHT_STAR_ENTERPRISE_AD00040="ARABIAN EXTRA LIGHT, STAR ENTERPRISE (AD00040)"
    ARABIAN_EXTRA_LIGHT_ARAMCO_AD00041="ARABIAN EXTRA LIGHT, ARAMCO (AD00041)"
    ARABIAN_HEAVY_AD00042="ARABIAN HEAVY (AD00042)"
    ARABIAN_HEAVY_CITGO_AD00044="ARABIAN HEAVY, CITGO (AD00044)"
    ARABIAN_HEAVY_EXXON_AD00046="ARABIAN HEAVY, EXXON (AD00046)"
    ARABIAN_HEAVY_AMOCO_AD00047="ARABIAN HEAVY, AMOCO (AD00047)"
    ARABIAN_HEAVY_STAR_ENTERPRISE_AD00049="ARABIAN HEAVY, STAR ENTERPRISE (AD00049)"
    ARABIAN_HEAVY_ARAMCO_AD00050="ARABIAN HEAVY, ARAMCO (AD00050)"
    ARABIAN_LIGHT_CITGO_AD00053="ARABIAN LIGHT, CITGO (AD00053)"
    ARABIAN_LIGHT_PHILLIPS_AD00055="ARABIAN LIGHT, PHILLIPS (AD00055)"
    ARABIAN_LIGHT_STAR_ENTERPRISE_AD00057="ARABIAN LIGHT, STAR ENTERPRISE (AD00057)"
    ARABIAN_LIGHT_ARAMCO_AD00058="ARABIAN LIGHT, ARAMCO (AD00058)"
    ARABIAN_MEDIUM_EXXON_AD00062="ARABIAN MEDIUM, EXXON (AD00062)"
    ARABIAN_MEDIUM_PHILLIPS_AD00063="ARABIAN MEDIUM, PHILLIPS (AD00063)"
    ARABIAN_MEDIUM_AMOCO_AD00064="ARABIAN MEDIUM, AMOCO (AD00064)"
    ARABIAN_MEDIUM_STAR_ENTERPRISE_AD00065="ARABIAN MEDIUM, STAR ENTERPRISE (AD00065)"
    ARABIAN_MEDIUM_CHEVRON_AD00066="ARABIAN MEDIUM, CHEVRON (AD00066)"
    ARGYL_AD00070="ARGYL (AD00070)"
    ATKINSON_AD00080="ATKINSON (AD00080)"
    AUTOMOTIVE_GASOLINE_EXXON_AD00084="AUTOMOTIVE GASOLINE, EXXON (AD00084)"
    AVALON_AD00085="AVALON (AD00085)"
    AVIATION_GASOLINE_100LL_STAR_ENTERPRISE_AD00092="AVIATION GASOLINE 100LL, STAR ENTERPRISE (AD00092)"
    AVIATION_GASOLINE_80_AD00094="AVIATION GASOLINE 80 (AD00094)"
    BACHAGUERO_CITGO_AD00095="BACHAGUERO, CITGO (AD00095)"
    BACHAQUERO_17_EXXON_AD00099="BACHAQUERO 17, EXXON (AD00099)"
    BACHEQUERO_HEAVY_AD00100="BACHEQUERO HEAVY (AD00100)"
    BACHEQUERO_MEDIUM_AD00101="BACHEQUERO MEDIUM (AD00101)"
    BAHIA_AD00102="BAHIA (AD00102)"
    BAKR_AD00103="BAKR (AD00103)"
    BANOCO_ABU_SAFAH_ARAMCO_AD00105="BANOCO ABU SAFAH, ARAMCO (AD00105)"
    BASRAH_AD00109="BASRAH (AD00109)"
    BASRAH_EXXON_AD00110="BASRAH, EXXON (AD00110)"
    BASS_STRAIT_AD00115="BASS STRAIT (AD00115)"
    BCF_13_AD00121="BCF 13 (AD00121)"
    BCF_17_AD00122="BCF 17 (AD00122)"
    BCF_22_CITGO_AD00124="BCF 22, CITGO (AD00124)"
    BCF_17_AMOCO_AD00127="BCF 17, AMOCO (AD00127)"
    BELAYIM_MARINE_AD00132="BELAYIM (MARINE) (AD00132)"
    BELAYIM_LAND_AD00133="BELAYIM (LAND) (AD00133)"
    BENT_HORN_A_02_AD00138="BENT HORN A-02 (AD00138)"
    BERRI_A_21_ARAMCO_AD00142="BERRI A-21, ARAMCO (AD00142)"
    BERYL_AD00143="BERYL (AD00143)"
    BFC_219_CITGO_AD00147="BFC 21.9, CITGO (AD00147)"
    BONNY_LIGHT_CITGO_AD00159="BONNY LIGHT, CITGO (AD00159)"
    BONNY_MEDIUM_CITGO_AD00162="BONNY MEDIUM, CITGO (AD00162)"
    BONNY_MEDIUM_AMOCO_AD00163="BONNY MEDIUM, AMOCO (AD00163)"
    BORHOLLA_AD00165="BORHOLLA (AD00165)"
    BOSCAN_AMOCO_AD00171="BOSCAN, AMOCO (AD00171)"
    BOW_RIVER_BLENDED_AD00174="BOW RIVER BLENDED (AD00174)"
    BRASS_RIVER_CITGO_AD00179="BRASS RIVER, CITGO (AD00179)"
    BRASS_RIVER_PHILLIPS_AD00181="BRASS RIVER, PHILLIPS (AD00181)"
    BREGA_ARCO_AD00185="BREGA, ARCO (AD00185)"
    BRENT_AD00187="BRENT (AD00187)"
    BRENT_CITGO_AD00189="BRENT, CITGO (AD00189)"
    BRENT_PHILLIPS_AD00190="BRENT, PHILLIPS (AD00190)"
    BRENT_MIX_EXXON_AD00196="BRENT MIX, EXXON (AD00196)"
    BRENT_SPAR_AD00197="BRENT SPAR (AD00197)"
    BUCHAN_AD00204="BUCHAN (AD00204)"
    BUNKER_C_FUEL_OIL_AD00208="BUNKER C FUEL OIL (AD00208)"
    CABINDA_CITGO_AD00213="CABINDA, CITGO (AD00213)"
    CABINDA_PHILLIPS_AD00215="CABINDA, PHILLIPS (AD00215)"
    CAMAR_AD00224="CAMAR (AD00224)"
    CANDON_SEC_PHILLIPS_AD00226="CANDON SEC, PHILLIPS (AD00226)"
    CANO_LIMON_CITGO_AD00227="CANO LIMON, CITGO (AD00227)"
    CANO_LIMON_PHILLIPS_AD00228="CANO LIMON, PHILLIPS (AD00228)"
    COBAN_BLEND_AD00254="COBAN BLEND (AD00254)"
    COBAN_BLEND_PHILLIPS_AD00255="COBAN BLEND, PHILLIPS (AD00255)"
    COHASSET_AD00257="COHASSET (AD00257)"
    COLD_LAKE_EXXON_AD00259="COLD LAKE, EXXON (AD00259)"
    COLD_LAKE_BLEND_ESSO_AD00262="COLD LAKE BLEND, ESSO (AD00262)"
    COLD_LAKE_DILUENT_ESSO_AD00263="COLD LAKE DILUENT, ESSO (AD00263)"
    COOK_INLET_DRIFT_RIVER_TERMINAL_AD00269="COOK INLET, DRIFT RIVER TERMINAL (AD00269)"
    CORMORANT_AD00270="CORMORANT (AD00270)"
    UNION_UNOCAL_AD00279="UNION, UNOCAL (AD00279)"
    CYRUS_ITOPF_AD00284="CYRUS, ITOPF (AD00284)"
    DANMARK_AD00289="DANMARK (AD00289)"
    DF2_SUMMER_DIESEL_TESORO_AD00293="DF2 SUMMER (DIESEL), TESORO (AD00293)"
    DF2_WINTER_DIESEL_TESORO_AD00294="DF2 WINTER (DIESEL), TESORO (AD00294)"
    DIESEL_AD00297="DIESEL (AD00297)"
    DJENO_PHILLIPS_AD00301="DJENO, PHILLIPS (AD00301)"
    DUNLIN_AD00315="DUNLIN (AD00315)"
    DURI_PHILLIPS_AD00316="DURI, PHILLIPS (AD00316)"
    EAST_TEXAS_AD00319="EAST TEXAS (AD00319)"
    EC_195_CONDENSATE_PHILLIPS_AD00322="EC 195-CONDENSATE, PHILLIPS (AD00322)"
    EKOFISK_AD00328="EKOFISK (AD00328)"
    EKOFISK_CITGO_AD00329="EKOFISK, CITGO (AD00329)"
    EKOFISK_EXXON_AD00332="EKOFISK, EXXON (AD00332)"
    EKOFISK_PHILLIPS_AD00333="EKOFISK, PHILLIPS (AD00333)"
    ELECTRICAL_INSULATING_OIL_VIRGIN_AD00346="ELECTRICAL INSULATING OIL (VIRGIN) (AD00346)"
    EMPIRE_ISLAND_AMOCO_AD00354="EMPIRE ISLAND, AMOCO (AD00354)"
    ESCALANTE_PHILLIPS_AD00362="ESCALANTE, PHILLIPS (AD00362)"
    ESCRAVOS_AMOCO_AD00365="ESCRAVOS, AMOCO (AD00365)"
    ESCRAVOS_CHEVRON_AD00366="ESCRAVOS, CHEVRON (AD00366)"
    FAO_CITGO_AD00376="FAO, CITGO (AD00376)"
    FCC_HEAVY_CYCLE_OIL_AD00377="FCC HEAVY CYCLE OIL (AD00377)"
    FEDERATED_AD00379="FEDERATED (AD00379)"
    FLOTTA_CITGO_AD00383="FLOTTA, CITGO (AD00383)"
    FLOTTA_AD00384="FLOTTA (AD00384)"
    FLOTTA_PHILLIPS_AD00385="FLOTTA, PHILLIPS (AD00385)"
    FLOTTA_MIX_AD00386="FLOTTA MIX (AD00386)"
    FORCADOS_CITGO_AD00388="FORCADOS, CITGO (AD00388)"
    FORCADOS_AMOCO_AD00389="FORCADOS, AMOCO (AD00389)"
    FORKED_ISLAND_TERMINAL_AMOCO_AD00391="FORKED ISLAND TERMINAL, AMOCO (AD00391)"
    FORTIES_AD00393="FORTIES (AD00393)"
    FOSTERTON_AD00397="FOSTERTON (AD00397)"
    FUEL_OIL_NO1_AVJET_A_STAR_ENTERPRISE_AD00403="FUEL OIL NO.1 (AVJET A), STAR ENTERPRISE (AD00403)"
    FUEL_OIL_NO1_DIESELHEATING_FUEL_PETRO_STAR_AD00404="FUEL OIL NO.1 (DIESEL/HEATING FUEL), PETRO STAR (AD00404)"
    FUEL_OIL_NO1_JET_FUEL_A_AD00412="FUEL OIL NO.1 (JET FUEL A) (AD00412)"
    FUEL_OIL_NO1_JET_FUEL_A_1_AD00413="FUEL OIL NO.1 (JET FUEL A-1) (AD00413)"
    FUEL_OIL_NO1_JET_FUEL_B_AD00414="FUEL OIL NO.1 (JET FUEL B) (AD00414)"
    FUEL_OIL_NO1_KEROSENE__AD00416="FUEL OIL NO.1 (KEROSENE)  (AD00416)"
    FUEL_OIL_NO2_DIESEL_STAR_ENTERPRISE_AD00431="FUEL OIL NO.2 (DIESEL), STAR ENTERPRISE (AD00431)"
    FUEL_OIL_NO2_HODIESEL_EXXON_AD00433="FUEL OIL NO.2 (HO/DIESEL), EXXON (AD00433)"
    FURRIAL_CITGO_AD00448="FURRIAL, CITGO (AD00448)"
    FURRIALMESA_28_EXXON_AD00449="FURRIAL/MESA 28, EXXON (AD00449)"
    GIPPSLAND_EXXON_AD00486="GIPPSLAND, EXXON (AD00486)"
    GIPPSLAND_MIX_ITOPF_AD00487="GIPPSLAND MIX, ITOPF (AD00487)"
    GUAFITA_CITGO_AD00506="GUAFITA, CITGO (AD00506)"
    GULF_OF_SUEZ_PHILLIPS_AD00513="GULF OF SUEZ, PHILLIPS (AD00513)"
    GULLFAKS_EXXON_AD00516="GULLFAKS, EXXON (AD00516)"
    HEAVY_LAKE_MIX_AD00530="HEAVY LAKE MIX (AD00530)"
    HEAVY_REFORMATE_AD00531="HEAVY REFORMATE (AD00531)"
    HI_317_PHILLIPS_AD00534="HI 317, PHILLIPS (AD00534)"
    HI_330349_CONDENSATE_PHILLIPS_AD00535="HI 330/349 CONDENSATE, PHILLIPS (AD00535)"
    HI_561_GRAND_CHENIER_PHILLIPS_AD00536="HI 561-GRAND CHENIER, PHILLIPS (AD00536)"
    HI_A_310_BCONDENSATE_PHILLIPS_AD00537="HI A-310-B/CONDENSATE, PHILLIPS (AD00537)"
    HIGH_ISLAND_AMOCO_AD00540="HIGH ISLAND, AMOCO (AD00540)"
    HIGH_ISLAND_BLOCK_154_PHILLIPS_AD00541="HIGH ISLAND BLOCK 154, PHILLIPS (AD00541)"
    HUTTON_AD00554="HUTTON (AD00554)"
    INTERPROVINCIAL_AD00563="INTERPROVINCIAL (AD00563)"
    IPPL_LIGHT_SOUR_BLEND_AD00565="IPPL LIGHT SOUR BLEND (AD00565)"
    ISSUNGNAK_AD00573="ISSUNGNAK (AD00573)"
    ISTHMUS_CITGO_AD00575="ISTHMUS, CITGO (AD00575)"
    ISTHMUS_PHILLIPS_AD00577="ISTHMUS, PHILLIPS (AD00577)"
    ISTHMUSMAYA_BLEND_AD00579="ISTHMUS/MAYA BLEND (AD00579)"
    ISTHMUSREFORMACACTUS_API_AD00580="ISTHMUS/REFORMA/CACTUS, API (AD00580)"
    JOBO_AD00589="JOBO (AD00589)"
    JOBOMORICHAL_ITOPF_AD00590="JOBO/MORICHAL, ITOPF (AD00590)"
    KHAFJI_AD00602="KHAFJI (AD00602)"
    KIRKUK_AD00610="KIRKUK (AD00610)"
    KIRKUK_EXXON_AD00611="KIRKUK, EXXON (AD00611)"
    KOAKOAK_0_22_AD00615="KOAKOAK 0-22 (AD00615)"
    KOLE_MARINE_AMOCO_AD00619="KOLE MARINE, AMOCO (AD00619)"
    KOPANOAR_2I_44_AD00622="KOPANOAR 2I-44 (AD00622)"
    KOPANOAR_M_13_AD00623="KOPANOAR M-13 (AD00623)"
    KOPANOAR_M_13A_AD00624="KOPANOAR M-13A (AD00624)"
    KUPARUK_AD00625="KUPARUK (AD00625)"
    KUWAIT_ARCO_AD00630="KUWAIT, ARCO (AD00630)"
    KUWAIT_CRUDE_OIL_LITERATURE_VALUES_AD00631="KUWAIT CRUDE OIL (LITERATURE VALUES) (AD00631)"
    KUWAIT_LIGHT_PHILLIPS_AD00633="KUWAIT LIGHT, PHILLIPS (AD00633)"
    LA_ROSA_AD00638="LA ROSA (AD00638)"
    LAGO_MEDIO_AD00644="LAGO MEDIO (AD00644)"
    LAGO_TRECO_CITGO_AD00647="LAGO TRECO, CITGO (AD00647)"
    LAGOTRECO_AD00648="LAGOTRECO (AD00648)"
    LAGUNA_AD00649="LAGUNA (AD00649)"
    LAGUNA_CITGO_AD00650="LAGUNA, CITGO (AD00650)"
    LAGUNA_22_CITGO_AD00651="LAGUNA 22, CITGO (AD00651)"
    LAGUNA_BLEND_24_CITGO_AD00652="LAGUNA BLEND 24, CITGO (AD00652)"
    LALANG_AD00665="LALANG (AD00665)"
    LARG_TRECO_MEDIUM_CITGO_AD00667="LARG TRECO MEDIUM, CITGO (AD00667)"
    LEDUC_WOODBEND_AD00672="LEDUC WOODBEND (AD00672)"
    LEONA_CITGO_AD00674="LEONA, CITGO (AD00674)"
    LIGHT_SOUR_BLEND_AD00680="LIGHT SOUR BLEND (AD00680)"
    LIUHUA_AMOCO_AD00682="LIUHUA, AMOCO (AD00682)"
    LOKELE_CITGO_AD00685="LOKELE, CITGO (AD00685)"
    LOKELE_EXXON_AD00686="LOKELE, EXXON (AD00686)"
    LUBRICATING_OIL_AUTO_ENGINE_OIL_VIRGIN_AD00697="LUBRICATING OIL (AUTO ENGINE OIL, VIRGIN) (AD00697)"
    MARALAGO_22_CITGO_AD00716="MARALAGO 22, CITGO (AD00716)"
    MARGHAM_AD00717="MARGHAM (AD00717)"
    MARIB_PHILLIPS_AD00718="MARIB, PHILLIPS (AD00718)"
    MARINE_DIESEL_FUEL_OIL_AD00721="MARINE DIESEL FUEL OIL (AD00721)"
    MARJANZULUF_ARAMCO_AD00725="MARJAN/ZULUF, ARAMCO (AD00725)"
    MAYA_CITGO_AD00732="MAYA, CITGO (AD00732)"
    MAYA_EXXON_AD00734="MAYA, EXXON (AD00734)"
    MAYA_PHILLIPS_AD00735="MAYA, PHILLIPS (AD00735)"
    MAYA_AMOCO_AD00736="MAYA, AMOCO (AD00736)"
    MAYOGIAK_AD00738="MAYOGIAK (AD00738)"
    MCARTHUR_RIVER_AD00741="MCARTHUR RIVER (AD00741)"
    MENEMOTA_AD00748="MENEMOTA (AD00748)"
    MENEMOTA_CITGO_AD00750="MENEMOTA, CITGO (AD00750)"
    MESA_28_CITGO_AD00756="MESA 28, CITGO (AD00756)"
    MESA_30_CITGO_AD00757="MESA 30, CITGO (AD00757)"
    MIDDLE_GROUND_SHOAL_AD00760="MIDDLE GROUND SHOAL (AD00760)"
    MORICHAL_AD00778="MORICHAL (AD00778)"
    NEKTORALIK_K_59_AD00809="NEKTORALIK K-59 (AD00809)"
    NEKTORALIK_K_59A_AD00810="NEKTORALIK K-59A (AD00810)"
    NERLERK_AD00811="NERLERK (AD00811)"
    NERLERK_M_98B_AD00812="NERLERK M-98B (AD00812)"
    NIGERIAN_EXP_B1_AD00817="NIGERIAN EXP. B1 (AD00817)"
    NIGERIAN_LGT_G_AD00818="NIGERIAN LGT G (AD00818)"
    NIGERIAN_LGT_M_AD00819="NIGERIAN LGT M (AD00819)"
    NIGERIAN_LIGHT_AD00820="NIGERIAN LIGHT (AD00820)"
    NIGERIAN_MEDIUM_AD00823="NIGERIAN MEDIUM (AD00823)"
    NIKISKI_AD00824="NIKISKI (AD00824)"
    NINIAN_AD00825="NINIAN (AD00825)"
    NINIAN_CITGO_AD00827="NINIAN, CITGO (AD00827)"
    NORTH_EAST_TEXAS_AD00834="NORTH EAST TEXAS (AD00834)"
    NORTH_SLOPE_AD00836="NORTH SLOPE (AD00836)"
    NORTH_SLOPE_CITGO_AD00837="NORTH SLOPE, CITGO (AD00837)"
    NORTH_SLOPE_PHILLIPS_AD00838="NORTH SLOPE, PHILLIPS (AD00838)"
    NOWRUZ_AD00839="NOWRUZ (AD00839)"
    OGUENDJO_AMOCO_AD00846="OGUENDJO, AMOCO (AD00846)"
    OLMECA_CITGO_AD00849="OLMECA, CITGO (AD00849)"
    OMAN_AD00852="OMAN (AD00852)"
    OMAN_PHILLIPS_AD00853="OMAN, PHILLIPS (AD00853)"
    OQUENDJO_AD00855="OQUENDJO (AD00855)"
    ORIENTE_CITGO_AD00858="ORIENTE, CITGO (AD00858)"
    OSEBERG_AD00859="OSEBERG (AD00859)"
    OSEBERG_EXXON_AD00860="OSEBERG, EXXON (AD00860)"
    OSEBERG_PHILLIPS_AD00861="OSEBERG, PHILLIPS (AD00861)"
    PALANCA_AD00864="PALANCA (AD00864)"
    PANUCO_AD00868="PANUCO (AD00868)"
    PANUKE_AD00869="PANUKE (AD00869)"
    PARENTIS_AD00875="PARENTIS (AD00875)"
    PECAN_ISLAND_AMOCO_AD00880="PECAN ISLAND, AMOCO (AD00880)"
    PEMBINA_AD00882="PEMBINA (AD00882)"
    PILON_AD00893="PILON (AD00893)"
    PILON_CITGO_AD00894="PILON, CITGO (AD00894)"
    PILON_ANACO_WAX_CITGO_AD00896="PILON-ANACO WAX, CITGO (AD00896)"
    PIPER_AD00897="PIPER (AD00897)"
    PITAS_POINT_AD00898="PITAS POINT (AD00898)"
    PL_COMPOSITE_STAR_ENTERPRISE_AD00899="PL COMPOSITE, STAR ENTERPRISE (AD00899)"
    PLATFORM_B_AD00900="PLATFORM B (AD00900)"
    PREMIUM_UNLEADED_GASOLINE_STAR_ENTERPRISE_AD00913="PREMIUM UNLEADED GASOLINE, STAR ENTERPRISE (AD00913)"
    QUA_IBOE_PHILLIPS_AD00924="QUA IBOE, PHILLIPS (AD00924)"
    RAGUSA_AD00932="RAGUSA (AD00932)"
    RANGELAND_SOUTH_LIGHT_AND_MEDIUM_AD00935="RANGELAND-SOUTH LIGHT AND MEDIUM (AD00935)"
    RAS_LANUF_AD00937="RAS LANUF (AD00937)"
    RATNA_AD00938="RATNA (AD00938)"
    REDWATER_AD00940="REDWATER (AD00940)"
    RESIDUAL_FUEL_900_TESORO_AD00944="RESIDUAL FUEL 900, TESORO (AD00944)"
    RIO_ZULIA_AD00949="RIO ZULIA (AD00949)"
    SAN_JOACHIM_AD00964="SAN JOACHIM (AD00964)"
    SANTA_CRUZ_AD00971="SANTA CRUZ (AD00971)"
    SANTA_MARIA_AD00973="SANTA MARIA (AD00973)"
    SARIR_ITOPF_AD00980="SARIR, ITOPF (AD00980)"
    SCHOONEBEEK_AD00983="SCHOONEBEEK (AD00983)"
    SHARJAH_AD00995="SHARJAH (AD00995)"
    SHIP_SHOAL_133_PHILLIPS_AD00999="SHIP SHOAL 133, PHILLIPS (AD00999)"
    SIRTICA_AD01006="SIRTICA (AD01006)"
    SMI_147_PHILLIPS_AD01008="SMI 147, PHILLIPS (AD01008)"
    SMI_66_PHILLIPS_AD01009="SMI 66, PHILLIPS (AD01009)"
    SOUR_BLEND_AD01022="SOUR BLEND (AD01022)"
    SOUTH_LOUISIANA_AD01025="SOUTH LOUISIANA (AD01025)"
    SOUTH_WEST_TEXAS_LIGHT_AD01030="SOUTH WEST TEXAS LIGHT (AD01030)"
    SOYO_AD01031="SOYO (AD01031)"
    SUEZ_MIX_AD01046="SUEZ MIX (AD01046)"
    SUNNILAND_EXXON_AD01050="SUNNILAND, EXXON (AD01050)"
    SWEET_BLEND_AD01054="SWEET BLEND (AD01054)"
    TACHING_AD01058="TACHING (AD01058)"
    TACHING_AD01059="TACHING (AD01059)"
    TAKULA_API_AD01062="TAKULA, API (AD01062)"
    TAKULA_CITGO_AD01063="TAKULA, CITGO (AD01063)"
    TAKULA_CHEVRON_AD01064="TAKULA, CHEVRON (AD01064)"
    TARSIUT_AD01070="TARSIUT (AD01070)"
    TARSIUT_A_25_AD01071="TARSIUT A-25 (AD01071)"
    TERRA_NOVA_AD01076="TERRA NOVA (AD01076)"
    TERRA_NOVA_K_08_DST_1_AD01077="TERRA NOVA K-08 DST #1 (AD01077)"
    TERRA_NOVA_K_08_DST_2_AD01078="TERRA NOVA K-08 DST #2 (AD01078)"
    TERRA_NOVA_K_08_DST_3_AD01079="TERRA NOVA K-08 DST #3 (AD01079)"
    TERRA_NOVA_K_08_DST_4_AD01080="TERRA NOVA K-08 DST #4 (AD01080)"
    TEXAS_GULF_COAST_HEAVY_AD01081="TEXAS GULF COAST HEAVY (AD01081)"
    TEXAS_GULF_COAST_LIGHT_AD01082="TEXAS GULF COAST LIGHT (AD01082)"
    TEXTRACT_STAR_ENTERPRISE_AD01083="TEXTRACT, STAR ENTERPRISE (AD01083)"
    TIA_JUANA_AD01088="TIA JUANA (AD01088)"
    TIA_JUANA_LIGHT_CITGO_AD01094="TIA JUANA LIGHT, CITGO (AD01094)"
    TIA_JUANA_MEDIUM_AD01096="TIA JUANA MEDIUM (AD01096)"
    TIA_JUANA_MEDIUM_CITGO_AD01097="TIA JUANA MEDIUM, CITGO (AD01097)"
    TIA_JUANA_MEDIUM_ARCO_AD01098="TIA JUANA MEDIUM, ARCO (AD01098)"
    TIA_JUANA_PESADO_AD01100="TIA JUANA PESADO (AD01100)"
    TRADING_BAY_OFFSHORE_COOK_INLET_AD01118="TRADING BAY (OFFSHORE COOK INLET) (AD01118)"
    TRANSMOUNTAIN_BLEND_AD01119="TRANSMOUNTAIN BLEND (AD01119)"
    TRINIDAD_AD01121="TRINIDAD (AD01121)"
    ULA_AD01133="ULA (AD01133)"
    UMM_SHAIF_AD01134="UMM SHAIF (AD01134)"
    UMM_SHARIF_PHILLIPS_AD01135="UMM SHARIF, PHILLIPS (AD01135)"
    UNLEADED_INTERM_GASOLINE_STAR_ENTERPRISE_AD01137="UNLEADED INTERM GASOLINE, STAR ENTERPRISE (AD01137)"
    UPPER_ZAKUM_PHILLIPS_AD01139="UPPER ZAKUM, PHILLIPS (AD01139)"
    URAL_AD01140="URAL (AD01140)"
    UVILUK_AD01141="UVILUK (AD01141)"
    VENEZUELA_MIX_AD01147="VENEZUELA MIX (AD01147)"
    WAFRA_EOCENE_AD01155="WAFRA EOCENE (AD01155)"
    WC_BLOCK_45_BEACH_CONDENSATE_PHILLIPS_AD01162="WC BLOCK 45 BEACH-CONDENSATE, PHILLIPS (AD01162)"
    WEST_GENERAL_TEXAS_AD01171="WEST GENERAL TEXAS (AD01171)"
    WEST_NEDERLAND_AD01172="WEST NEDERLAND (AD01172)"
    WEST_TEXAS_ELLENBURGER_AD01175="WEST TEXAS ELLENBURGER (AD01175)"
    WEST_TEXAS_LIGHT_AD01177="WEST TEXAS LIGHT (AD01177)"
    WEYBURN_MIDALE_AD01180="WEYBURN-MIDALE (AD01180)"
    YANBU_ARABIAN_LIGHT_ARAMCO_AD01184="YANBU ARABIAN LIGHT, ARAMCO (AD01184)"
    YOMBO_AMOCO_AD01186="YOMBO, AMOCO (AD01186)"
    ZAIRE_API_AD01189="ZAIRE, API (AD01189)"
    ZAIRE_CHEVRON_AD01191="ZAIRE, CHEVRON (AD01191)"
    ZAKUA_AD01193="ZAKUA (AD01193)"
    ZAKUM_AD01194="ZAKUM (AD01194)"
    ZETA_NORTH_AD01200="ZETA NORTH (AD01200)"
    MARINE_DIESEL_F_76_MANCHESTER_FUEL_AD01215="MARINE DIESEL F-76, MANCHESTER FUEL (AD01215)"
    KERN_COUNTY_BLEND_AD01217="KERN COUNTY BLEND (AD01217)"
    VENEZUELA_RECON_AD01219="VENEZUELA RECON (AD01219)"
    DAQIN_AD01220="DAQIN (AD01220)"
    SHIAN_LI_AD01221="SHIAN LI (AD01221)"
    HUIZHOU_AD01222="HUIZHOU (AD01222)"
    WEST_TEXAS_INTERMEDIATE_OIL__GAS_AD01223="WEST TEXAS INTERMEDIATE, OIL & GAS (AD01223)"
    MAIN_PASS_140_PENNZOIL_AD01225="MAIN PASS 140, PENNZOIL (AD01225)"
    JABIRU_BHP_PETROLEUM_AD01232="JABIRU, BHP PETROLEUM (AD01232)"
    JABIRU_1A_BHP_PETROLEUM_AD01233="JABIRU 1A, BHP PETROLEUM (AD01233)"
    KUTUBU_LIGHT_BHP_PETROLEUM_AD01235="KUTUBU LIGHT, BHP PETROLEUM (AD01235)"
    GIPPSLAND_BHP_PETROLEUM_AD01236="GIPPSLAND, BHP PETROLEUM (AD01236)"
    UDANG_OIL__GAS_AD01258="UDANG, OIL & GAS (AD01258)"
    DURI_OIL__GAS_AD01262="DURI, OIL & GAS (AD01262)"
    SOVIET_EXPORT_OIL__GAS_AD01264="SOVIET EXPORT, OIL & GAS (AD01264)"
    BARROW_ISLAND_OIL__GAS_AD01269="BARROW ISLAND, OIL & GAS (AD01269)"
    BELAYIM_OIL__GAS_AD01301="BELAYIM, OIL & GAS (AD01301)"
    SHARJAH_OIL__GAS_AD01326="SHARJAH, OIL & GAS (AD01326)"
    LLOYDMINSTER_OIL__GAS_AD01345="LLOYDMINSTER, OIL & GAS (AD01345)"
    ALASKA_NORTH_SLOPE_OIL__GAS_AD01346="ALASKA NORTH SLOPE, OIL & GAS (AD01346)"
    STATJORD_OIL__GAS_AD01357="STATJORD, OIL & GAS (AD01357)"
    FLOTTA_OIL__GAS_AD01369="FLOTTA, OIL & GAS (AD01369)"
    FORTIES_OIL__GAS_AD01378="FORTIES, OIL & GAS (AD01378)"
    ESCRAVOS_OIL__GAS_AD01392="ESCRAVOS, OIL & GAS (AD01392)"
    BELIDA_OIL__GAS_AD01401="BELIDA, OIL & GAS (AD01401)"
    RABBI_COASTAL_EAGLE_POINT_OIL_AD01411="RABBI, COASTAL EAGLE POINT OIL (AD01411)"
    SOLVENT_NEUTRAL_OIL_320_STAR_ENTERPRISE_AD01412="SOLVENT NEUTRAL OIL 320, STAR ENTERPRISE (AD01412)"
    ROSSIIELF_RUSSIAN_JOINT_STOCK_CO_AD01413="ROSSIIELF, RUSSIAN JOINT STOCK CO (AD01413)"
    KUTUBU_AMSA_AD01419="KUTUBU, AMSA (AD01419)"
    GRIFFIN_AMSA_AD01420="GRIFFIN, AMSA (AD01420)"
    NSW_CONDENSATE_AMSA_AD01421="NSW CONDENSATE, AMSA (AD01421)"
    KABINDA_GALLAGER_MARINE_AD01424="KABINDA, GALLAGER MARINE (AD01424)"
    NEMBA_GALLAGER_MARINE_AD01425="NEMBA, GALLAGER MARINE (AD01425)"
    FUEL_OIL_NO2_AMOCO_AD01427="FUEL OIL NO.2, AMOCO (AD01427)"
    TEAK_AND_SAMAAN_AMOCO_AD01428="TEAK AND SAMAAN, AMOCO (AD01428)"
    GALEOTA_MIX_AMOCO_AD01429="GALEOTA MIX, AMOCO (AD01429)"
    POUI_AMOCO_AD01430="POUI, AMOCO (AD01430)"
    QATARDUKHAM_CHEVRON_AD01432="QATAR/DUKHAM, CHEVRON (AD01432)"
    ALGERIAN_CONDENSATE_SHELL_OIL_AD01433="ALGERIAN CONDENSATE, SHELL OIL (AD01433)"
    ARABIAN_MEDIUM_SHELL_OIL_AD01434="ARABIAN MEDIUM, SHELL OIL (AD01434)"
    ARUN_CONDENSATE_SHELL_OIL_AD01435="ARUN CONDENSATE, SHELL OIL (AD01435)"
    BACHAQUERO_SHELL_OIL_AD01436="BACHAQUERO, SHELL OIL (AD01436)"
    BADAK_SHELL_OIL_AD01437="BADAK, SHELL OIL (AD01437)"
    BETA_PRODUCTION_SHELL_OIL_AD01438="BETA PRODUCTION, SHELL OIL (AD01438)"
    BONITO_PL_SOUR_SHELL_OIL_AD01439="BONITO P/L SOUR, SHELL OIL (AD01439)"
    BONNY_LIGHT_SHELL_OIL_AD01440="BONNY LIGHT, SHELL OIL (AD01440)"
    BRASS_RIVER_SHELL_OIL_AD01441="BRASS RIVER, SHELL OIL (AD01441)"
    CABINDA_BLEND_SHELL_OIL_AD01442="CABINDA BLEND, SHELL OIL (AD01442)"
    COGNAC_BLOCK_194_SHELL_OIL_AD01443="COGNAC-BLOCK 194, SHELL OIL (AD01443)"
    DJENO_SHELL_OIL_AD01444="DJENO, SHELL OIL (AD01444)"
    ERAWAN_CONDENSATE_SHELL_OIL_AD01445="ERAWAN CONDENSATE, SHELL OIL (AD01445)"
    ESCRAVOS_SHELL_OIL_AD01446="ESCRAVOS, SHELL OIL (AD01446)"
    ETCHEGOIN_SHELL_OIL_AD01447="ETCHEGOIN, SHELL OIL (AD01447)"
    FLOTTA_SHELL_OIL_AD01448="FLOTTA, SHELL OIL (AD01448)"
    FORCADOS_SHELL_OIL_AD01449="FORCADOS, SHELL OIL (AD01449)"
    FORTIES_SHELL_OIL_AD01450="FORTIES, SHELL OIL (AD01450)"
    FURRIAL_SHELL_OIL_AD01451="FURRIAL, SHELL OIL (AD01451)"
    GIPPSLAND_SHELL_OIL_AD01452="GIPPSLAND, SHELL OIL (AD01452)"
    GREEN_CANYON_SHELL_OIL_AD01453="GREEN CANYON, SHELL OIL (AD01453)"
    GULLFAKS_SHELL_OIL_AD01454="GULLFAKS, SHELL OIL (AD01454)"
    HARDING_SHELL_OIL_AD01455="HARDING, SHELL OIL (AD01455)"
    HIGH_ISLAND_SWEET_SHELL_OIL_AD01456="HIGH ISLAND SWEET, SHELL OIL (AD01456)"
    HUNTINGTON_BEACH_SHELL_OIL_AD01457="HUNTINGTON BEACH, SHELL OIL (AD01457)"
    ISTHMUS_SHELL_OIL_AD01458="ISTHMUS, SHELL OIL (AD01458)"
    JABIRU_SHELL_OIL_AD01460="JABIRU, SHELL OIL (AD01460)"
    KERN_RIVER_SWEPI_SHELL_OIL_AD01461="KERN RIVER-SWEPI, SHELL OIL (AD01461)"
    KIRKUK_SHELL_OIL_AD01462="KIRKUK, SHELL OIL (AD01462)"
    KOLE_SHELL_OIL_AD01463="KOLE, SHELL OIL (AD01463)"
    KUTUBU_SHELL_OIL_AD01464="KUTUBU, SHELL OIL (AD01464)"
    LAGOCINCO_SHELL_OIL_AD01465="LAGOCINCO, SHELL OIL (AD01465)"
    LAGOMAR_SHELL_OIL_AD01466="LAGOMAR, SHELL OIL (AD01466)"
    LAGOTRECO_SHELL_OIL_AD01467="LAGOTRECO, SHELL OIL (AD01467)"
    LOKELE_SHELL_OIL_AD01468="LOKELE, SHELL OIL (AD01468)"
    LLOYDMINSTER_SHELL_OIL_AD01469="LLOYDMINSTER, SHELL OIL (AD01469)"
    ARABIAN_LIGHT_SHELL_OIL_AD01470="ARABIAN LIGHT, SHELL OIL (AD01470)"
    LORETO_SHELL_OIL_AD01471="LORETO, SHELL OIL (AD01471)"
    LUCINA_SHELL_OIL_AD01472="LUCINA, SHELL OIL (AD01472)"
    MAIN_PASS_49_CONDENSATE_SHELL_OIL_AD01473="MAIN PASS 49 CONDENSATE, SHELL OIL (AD01473)"
    MAYA_SHELL_OIL_AD01474="MAYA, SHELL OIL (AD01474)"
    MANDJI_SHELL_OIL_AD01475="MANDJI, SHELL OIL (AD01475)"
    MURBAN_SHELL_OIL_AD01476="MURBAN, SHELL OIL (AD01476)"
    OLMECA_SHELL_OIL_AD01477="OLMECA, SHELL OIL (AD01477)"
    OMAN_SHELL_OIL_AD01478="OMAN, SHELL OIL (AD01478)"
    ORIENTE_SHELL_OIL_AD01479="ORIENTE, SHELL OIL (AD01479)"
    OSEBERG_SHELL_OIL_AD01480="OSEBERG, SHELL OIL (AD01480)"
    PALANCA_SHELL_OIL_AD01481="PALANCA, SHELL OIL (AD01481)"
    PECAN_ISLAND_SHELL_OIL_AD01482="PECAN ISLAND, SHELL OIL (AD01482)"
    QUA_IBOE_SHELL_OIL_AD01483="QUA IBOE, SHELL OIL (AD01483)"
    RABI_BLEND_SHELL_OIL_AD01484="RABI BLEND, SHELL OIL (AD01484)"
    RABI_KOUNGA_SHELL_OIL_AD01485="RABI-KOUNGA, SHELL OIL (AD01485)"
    SAHARAN_BLEND_BEJAIA_SHELL_OIL_AD01486="SAHARAN BLEND BEJAIA, SHELL OIL (AD01486)"
    SAHARAN_BLEND_ARZEW_SHELL_OIL_AD01487="SAHARAN BLEND ARZEW, SHELL OIL (AD01487)"
    SKUA_SHELL_OIL_AD01488="SKUA, SHELL OIL (AD01488)"
    SOYO_SHELL_OIL_AD01489="SOYO, SHELL OIL (AD01489)"
    TIA_JUANA_LIGHT_SHELL_OIL_AD01490="TIA JUANA LIGHT, SHELL OIL (AD01490)"
    TIERRA_DEL_FUEGO_SHELL_OIL_AD01491="TIERRA DEL FUEGO, SHELL OIL (AD01491)"
    VENTURA_SHELL_TAYLOR_LEASE_SHELL_OIL_AD01492="VENTURA SHELL TAYLOR LEASE, SHELL OIL (AD01492)"
    VIOSCA_KNOLL_826_SHELL_OIL_AD01493="VIOSCA KNOLL 826, SHELL OIL (AD01493)"
    WEST_DELTA_BLOCK_89_SHELL_OIL_AD01494="WEST DELTA BLOCK 89, SHELL OIL (AD01494)"
    WEST_LAKE_VERRET_SHELL_OIL_AD01495="WEST LAKE VERRET, SHELL OIL (AD01495)"
    XIJIANG_SHELL_OIL_AD01496="XIJIANG, SHELL OIL (AD01496)"
    YORBA_LINDA_SHELL_SHELL_OIL_AD01497="YORBA LINDA SHELL, SHELL OIL (AD01497)"
    YOWLUMNE_SHELL_OIL_AD01498="YOWLUMNE, SHELL OIL (AD01498)"
    ZAIRE_SHELL_OIL_AD01499="ZAIRE, SHELL OIL (AD01499)"
    JET_A_1__MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01500="JET A-1,  MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01500)"
    DUAL_PURPOSE_KEROSINE__MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01501="DUAL PURPOSE KEROSINE,  MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01501)"
    MCKEE_BLEND_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01507="MCKEE BLEND, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01507)"
    MAUI_F_SAND_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01508="MAUI F SAND, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01508)"
    MCKEE_BLEND_50_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01510="MCKEE BLEND 50%, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01510)"
    MCKEE_BLEND_25_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01511="MCKEE BLEND 25%, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01511)"
    MCKEE_BLEND_10_NGAT_1_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01512="MCKEE BLEND 10% NGAT-1, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01512)"
    MCKEE_BLEND_10_NGAT_2_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01513="MCKEE BLEND 10% NGAT-2, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01513)"
    MCKEE_BLEND_10_NGAT_3_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01514="MCKEE BLEND 10% NGAT-3, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01514)"
    HANDIL_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01515="HANDIL, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01515)"
    BARROW_ISLAND_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01517="BARROW ISLAND, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01517)"
    BRASS_RIVER_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01519="BRASS RIVER, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01519)"
    DUBAI_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01520="DUBAI, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01520)"
    MURBAN_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01521="MURBAN, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01521)"
    MAUI_B_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01522="MAUI B, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01522)"
    KUTUBU_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01524="KUTUBU, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01524)"
    GRIFFIN_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01525="GRIFFIN, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01525)"
    MIRI_LIGHT_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01528="MIRI LIGHT, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01528)"
    SYNGAS_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01529="SYNGAS, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01529)"
    LABUAN_MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01530="LABUAN, MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01530)"
    OMAN__MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01533="OMAN,  MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01533)"
    THEVENARD__MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01535="THEVENARD,  MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01535)"
    WIDURI__MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01536="WIDURI,  MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01536)"
    KHAFJI__MARITIME_SAFETY_AUTHORITY_OF_NEW_ZEALAND_AD01537="KHAFJI,  MARITIME SAFETY AUTHORITY OF NEW ZEALAND (AD01537)"
    FORCADOS_BP_AD01552="FORCADOS, BP (AD01552)"
    WEST_TEXAS_SOUR_BP_AD01553="WEST TEXAS SOUR, BP (AD01553)"
    LIGHT_LOUISIANNA_SWEET_BP_AD01554="LIGHT LOUISIANNA SWEET, BP (AD01554)"
    RINCON_DE_LOS_SAUCES_OIL__GAS_AD01556="RINCON DE LOS SAUCES, OIL & GAS (AD01556)"
    MEDANITO_OIL__GAS_AD01557="MEDANITO, OIL & GAS (AD01557)"
    ESCRAVOS_SWAMP_BLEND_CHEVRON_AD01561="ESCRAVOS SWAMP BLEND, CHEVRON (AD01561)"
    BENIN_RIVER_CHEVRON_AD01562="BENIN RIVER, CHEVRON (AD01562)"
    NORTHWEST_CHARGE_STOCK_CHEVRON_AD01567="NORTHWEST CHARGE STOCK, CHEVRON (AD01567)"
    ARABIAN_EXTRA_LIGHT_BOUCHARD_AD01577="ARABIAN EXTRA LIGHT, BOUCHARD (AD01577)"
    BRENT_SUN_AD01579="BRENT, SUN (AD01579)"
    MONTEREY_TORCH_AD01581="MONTEREY, TORCH (AD01581)"
    ODUDU_EXXON_AD01585="ODUDU, EXXON (AD01585)"
    BELIDA_AD01612="BELIDA (AD01612)"
    BINTULU_AD01614="BINTULU (AD01614)"
    BUNKER_C_FUEL_OIL_ALASKA_AD01621="BUNKER C FUEL OIL (ALASKA) (AD01621)"
    BUNKER_C_FUEL_OIL_IRVING_WHALE_AD01622="BUNKER C FUEL OIL (IRVING WHALE) (AD01622)"
    CANO_LIMON_AD01627="CANO LIMON (AD01627)"
    CUSIANA_AD01634="CUSIANA (AD01634)"
    FCC_FEED_AD01652="FCC FEED (AD01652)"
    FCC_MEDIUM_CYCLE_OIL_AD01654="FCC MEDIUM CYCLE OIL (AD01654)"
    IFO_180_AD01676="IFO 180 (AD01676)"
    ORIENTE_AD01705="ORIENTE (AD01705)"
    ORIMULSION_AD01706="ORIMULSION (AD01706)"
    PENNINGTON_AD01707="PENNINGTON (AD01707)"
    SARIR_AD01726="SARIR (AD01726)"
    SIBERIAN_BLEND_AD01732="SIBERIAN BLEND (AD01732)"
    SOUTH_PASS_BLOCK_67_AD01739="SOUTH PASS BLOCK 67 (AD01739)"
    NEWFOUNDLAND_OFFSHORE_BURN_EXPERIMENT_AD01758="NEWFOUNDLAND OFFSHORE BURN EXPERIMENT (AD01758)"
    FUEL_OIL_NO1_JET_B_ALASKA_AD01765="FUEL OIL NO.1 (JET B, ALASKA) (AD01765)"
    DIESELHEATING_OIL_NO2_CHEVRON_AD01774="DIESEL/HEATING OIL NO.2, CHEVRON (AD01774)"
    DESTIN_DOME_CIS_MMS_AD01775="DESTIN DOME CIS, MMS (AD01775)"
    MOTOR_GASOLINE_PREMIUM_UNLEADED_SHELL_REFINING_PTY__AD01776="MOTOR GASOLINE-PREMIUM UNLEADED, SHELL REFINING PTY  (AD01776)"
    MOTOR_GASOLINE_UNLEADED_SHELL_REFINING_PTY__AD01777="MOTOR GASOLINE-UNLEADED, SHELL REFINING PTY  (AD01777)"
    MOTOR_GASOLINE_LEADED_SHELL_REFINING_PTY__AD01778="MOTOR GASOLINE-LEADED, SHELL REFINING PTY  (AD01778)"
    AUTOMOTIVE_DIESEL_FUEL_SHELL_REFINING_PTY__AD01779="AUTOMOTIVE DIESEL FUEL, SHELL REFINING PTY  (AD01779)"
    AVIATION_TURBINE_FUEL_SHELL_REFINING_PTY__AD01786="AVIATION TURBINE FUEL, SHELL REFINING PTY  (AD01786)"
    WHITE_SPIRIT_SHELL_REFINING_PTY__AD01800="WHITE SPIRIT, SHELL REFINING PTY  (AD01800)"
    NKOSSA_SHELL_REFINING_PTY__AD01804="NKOSSA, SHELL REFINING PTY  (AD01804)"
    MURBAN_SHELL_REFINING_PTY__AD01805="MURBAN, SHELL REFINING PTY  (AD01805)"
    OMAN_SHELL_REFINING_PTY__AD01806="OMAN, SHELL REFINING PTY  (AD01806)"
    GIPPSLAND_SHELL_REFINING_PTY__AD01809="GIPPSLAND, SHELL REFINING PTY  (AD01809)"
    THEVENARD_SHELL_REFINING_PTY__AD01811="THEVENARD, SHELL REFINING PTY  (AD01811)"
    XI_JANG_SHELL_REFINING_PTY__AD01812="XI-JANG, SHELL REFINING PTY  (AD01812)"
    ATTAKA_SHELL_REFINING_PTY__AD01813="ATTAKA, SHELL REFINING PTY  (AD01813)"
    ARDJUNA_SHELL_REFINING_PTY__AD01814="ARDJUNA, SHELL REFINING PTY  (AD01814)"
    CINTA_SHELL_REFINING_PTY__AD01815="CINTA, SHELL REFINING PTY  (AD01815)"
    WIDURI_SHELL_REFINING_PTY__AD01816="WIDURI, SHELL REFINING PTY  (AD01816)"
    LALANG_SHELL_REFINING_PTY__AD01817="LALANG, SHELL REFINING PTY  (AD01817)"
    MINAS_SHELL_REFINING_PTY__AD01818="MINAS, SHELL REFINING PTY  (AD01818)"
    MAUI_SHELL_REFINING_PTY__AD01819="MAUI, SHELL REFINING PTY  (AD01819)"
    MCKEE_SHELL_REFINING_PTY__AD01820="MCKEE, SHELL REFINING PTY  (AD01820)"
    BACH_HO_SHELL_REFINING_PTY__AD01822="BACH HO, SHELL REFINING PTY  (AD01822)"
    CHALLIS_BHP_PETROLEUM_AD01823="CHALLIS, BHP PETROLEUM (AD01823)"
    GRIFFIN_BHP_PETROLEUM_AD01824="GRIFFIN, BHP PETROLEUM (AD01824)"
    HARRIET_APACHE_ENERGY_LTD_AD01826="HARRIET, APACHE ENERGY LTD (AD01826)"
    STAG_APACHE_ENERGY_LTD_AD01827="STAG, APACHE ENERGY LTD (AD01827)"
    COOPER_BASIN_SANTOS_LTD_AD01830="COOPER BASIN, SANTOS LTD (AD01830)"
    GIPPSLAND_AMSA_AD01834="GIPPSLAND, AMSA (AD01834)"
    ALASKA_NORTH_SLOPE_PUMP_STATION_9_BP_AD01850="ALASKA NORTH SLOPE-PUMP STATION #9, BP (AD01850)"
    QATAR_NORTH_FIELD_CONDENSATE_NFR_1_MOBIL_AD01851="QATAR NORTH FIELD CONDENSATE (NFR-1), MOBIL (AD01851)"
    AIRILE_BP_AD01853="AIRILE, BP (AD01853)"
    BARROW_BP_AD01854="BARROW, BP (AD01854)"
    BLINA_BP_AD01855="BLINA, BP (AD01855)"
    JACKSON_BP_AD01856="JACKSON, BP (AD01856)"
    SURAT_BASIN_BP_AD01857="SURAT BASIN, BP (AD01857)"
    THEVENAND_BP_AD01858="THEVENAND, BP (AD01858)"
    VARANUS_BP_AD01859="VARANUS, BP (AD01859)"
    WANDO_BP_AD01860="WANDO, BP (AD01860)"
    UMM_SHAIF_BP_AD01861="UMM SHAIF, BP (AD01861)"
    UPPER_ZAKUM_BP_AD01862="UPPER ZAKUM, BP (AD01862)"
    MARGHAM_BP_AD01863="MARGHAM, BP (AD01863)"
    KUWAIT_BP_AD01864="KUWAIT, BP (AD01864)"
    KHAFJI_BP_AD01865="KHAFJI, BP (AD01865)"
    AL_RAYYAN_BP_AD01866="AL RAYYAN, BP (AD01866)"
    SAJAA_CONDENSATE_BP_AD01868="SAJAA CONDENSATE, BP (AD01868)"
    NANNAI_LIGHT_BP_AD01869="NANNAI LIGHT, BP (AD01869)"
    BELIDA_BP_AD01870="BELIDA, BP (AD01870)"
    BONTANG_MIX_BP_AD01872="BONTANG MIX, BP (AD01872)"
    HANDIL_BP_AD01873="HANDIL, BP (AD01873)"
    KERAPU_BP_AD01874="KERAPU, BP (AD01874)"
    MIRI_LIGHT_BP_AD01876="MIRI LIGHT, BP (AD01876)"
    CHERVIL_NOVUS_WA_PTY_LTD_AD01877="CHERVIL, NOVUS WA PTY LTD (AD01877)"
    ARABIAN_EXTRA_LIGHT_MOBIL_OIL_AUSTRALIA_AD01882="ARABIAN EXTRA LIGHT, MOBIL OIL AUSTRALIA (AD01882)"
    BASRAH_LIGHT_MOBIL_OIL_AUSTRALIA_AD01884="BASRAH LIGHT, MOBIL OIL AUSTRALIA (AD01884)"
    BELIDA_MOBIL_OIL_AUSTRALIA__AD01885="BELIDA, MOBIL OIL AUSTRALIA  (AD01885)"
    EAST_SPAB_MOBIL_OIL_AUSTRALIA_AD01887="EAST SPAB, MOBIL OIL AUSTRALIA (AD01887)"
    ERAWAN_MOBIL_OIL_AUSTRALIA__AD01888="ERAWAN, MOBIL OIL AUSTRALIA  (AD01888)"
    KUTUBU_LIGHT_MOBIL_OIL_AUSTRALIA__AD01889="KUTUBU LIGHT, MOBIL OIL AUSTRALIA  (AD01889)"
    QATAR_LAND_MOBIL_OIL_AUSTRALIA__AD01891="QATAR LAND, MOBIL OIL AUSTRALIA  (AD01891)"
    QATAR_MARINE_MOBIL_OIL_AUSTRALIA_AD01892="QATAR MARINE, MOBIL OIL AUSTRALIA (AD01892)"
    THAMMAMA_MOBIL_OIL_AUSTRALIA__AD01893="THAMMAMA, MOBIL OIL AUSTRALIA  (AD01893)"
    UPPER_ZAKUM_MOBIL_OIL_AUSTRALIA__AD01894="UPPER ZAKUM, MOBIL OIL AUSTRALIA  (AD01894)"
    WANDOO_MOBIL_OIL_AUSTRALIA_AD01895="WANDOO, MOBIL OIL AUSTRALIA (AD01895)"
    BELIDA_CALTEX_AD01896="BELIDA, CALTEX (AD01896)"
    KUKAPU_CALTEX_AD01897="KUKAPU, CALTEX (AD01897)"
    BEKOPAI_CALTEX_AD01898="BEKOPAI, CALTEX (AD01898)"
    SENIPAH_CALTEX_AD01899="SENIPAH, CALTEX (AD01899)"
    IMA_CALTEX_AD01900="IMA, CALTEX (AD01900)"
    ORIENTE_OIL__GAS_AD01904="ORIENTE, OIL & GAS (AD01904)"
    MAYA_OIL__GAS_AD01906="MAYA, OIL & GAS (AD01906)"
    OLMECA_OIL__GAS_AD01907="OLMECA, OIL & GAS (AD01907)"
    BOSCAN_OIL__GAS_AD01911="BOSCAN, OIL & GAS (AD01911)"
    LA_ROSA_MEDIUM_OIL__GAS_AD01912="LA ROSA MEDIUM, OIL & GAS (AD01912)"
    MEREY_OIL__GAS_AD01913="MEREY, OIL & GAS (AD01913)"
    MURBAN_OIL__GAS_AD01916="MURBAN, OIL & GAS (AD01916)"
    BASRAH_OIL__GAS_AD01922="BASRAH, OIL & GAS (AD01922)"
    KIRKUK_OIL__GAS_AD01924="KIRKUK, OIL & GAS (AD01924)"
    KUWAIT_EXPORT_OIL__GAS_AD01925="KUWAIT EXPORT, OIL & GAS (AD01925)"
    ARABIAN_LIGHT_OIL__GAS_AD01927="ARABIAN LIGHT, OIL & GAS (AD01927)"
    SEPINGGAN_YAKIN_MIXED_OIL__GAS_AD01929="SEPINGGAN-YAKIN MIXED, OIL & GAS (AD01929)"
    SAHARAN_BLEND_OIL__GAS_AD01930="SAHARAN BLEND, OIL & GAS (AD01930)"
    RAINBOW_LIGHT_AND_MEDIUM_OIL__GAS_AD01938="RAINBOW LIGHT AND MEDIUM, OIL & GAS (AD01938)"
    RANGELAND_SOUTH_OIL__GAS_AD01939="RANGELAND-SOUTH, OIL & GAS (AD01939)"
    EKOFISK_OIL__GAS_AD01944="EKOFISK, OIL & GAS (AD01944)"
    GULLFAKS_OIL__GAS_AD01945="GULLFAKS, OIL & GAS (AD01945)"
    OSEBERG_OIL__GAS_AD01946="OSEBERG, OIL & GAS (AD01946)"
    ARGYLL_OIL__GAS_AD01947="ARGYLL, OIL & GAS (AD01947)"
    BRENT_OIL__GAS_AD01950="BRENT, OIL & GAS (AD01950)"
    MIX_GEISUM_GEISUM_OIL_AD01970="MIX GEISUM, GEISUM OIL (AD01970)"
    NORTH_GEISUM_GEISUM_OIL_AD01971="NORTH GEISUM, GEISUM OIL (AD01971)"
    TAWILA_GEISUM_OIL_AD01972="TAWILA, GEISUM OIL (AD01972)"
    SOUTH_GEISUM_GEISUM_OIL_AD01973="SOUTH GEISUM, GEISUM OIL (AD01973)"
    VIOSCA_KNOLL_BLOCK_990_AD01978="VIOSCA KNOLL BLOCK 990 (AD01978)"
    POSIDEN_EQUILON_AD01981="POSIDEN, EQUILON (AD01981)"
    ABOOZAR_AD01983="ABOOZAR (AD01983)"
    ABU_AL_BU_KHOOSH_AD01984="ABU AL BU KHOOSH (AD01984)"
    ADGO_AD01985="ADGO (AD01985)"
    ALASKA_NORTH_SLOPE_AD01986="ALASKA NORTH SLOPE (AD01986)"
    ALASKA_NORTH_SLOPE_MIDDLE_PIPELINE_AD01987="ALASKA NORTH SLOPE (MIDDLE PIPELINE) (AD01987)"
    ALASKA_NORTH_SLOPE_NORTHERN_PIPELINE_AD01988="ALASKA NORTH SLOPE (NORTHERN PIPELINE) (AD01988)"
    ALASKA_NORTH_SLOPE_SOCSEX_AD01989="ALASKA NORTH SLOPE (SOCSEX) (AD01989)"
    ALASKA_NORTH_SLOPE_SOUTHERN_PIPELINE_AD01990="ALASKA NORTH SLOPE (SOUTHERN PIPELINE) (AD01990)"
    ALBA_AD01991="ALBA (AD01991)"
    ALBERTA_SWEET_MIXED_BLEND_PETAWAWA_AD01993="ALBERTA SWEET MIXED BLEND (PETAWAWA) (AD01993)"
    ALBERTA_SWEET_MIXED_BLEND_REFERENCE_2_AD01994="ALBERTA SWEET MIXED BLEND (REFERENCE #2) (AD01994)"
    ALBERTA_SWEET_MIXED_BLEND_REFERENCE_3_AD01995="ALBERTA SWEET MIXED BLEND (REFERENCE #3) (AD01995)"
    AMAULIGAK_AD01998="AMAULIGAK (AD01998)"
    AMNA_AD01999="AMNA (AD01999)"
    ARABIAN_AD02000="ARABIAN (AD02000)"
    ARABIAN_LIGHT_AD02002="ARABIAN LIGHT (AD02002)"
    ARABIAN_MEDIUM_AD02003="ARABIAN MEDIUM (AD02003)"
    ARIMBI_AD02006="ARIMBI (AD02006)"
    ASHTART_AD02007="ASHTART (AD02007)"
    ATTAKA_AD02012="ATTAKA (AD02012)"
    AVALON_AD02014="AVALON (AD02014)"
    AVIATION_GASOLINE_100_AD02015="AVIATION GASOLINE 100 (AD02015)"
    BACH_HO_AD02018="BACH HO (AD02018)"
    BACHAQUERO_AD02019="BACHAQUERO (AD02019)"
    BADAK_AD02020="BADAK (AD02020)"
    BAHRGANSARNOWRUZ_AD02021="BAHRGANSAR/NOWRUZ (AD02021)"
    BARROW_ISLAND_AD02022="BARROW ISLAND (AD02022)"
    BASRAH_HEAVY_AD02023="BASRAH HEAVY (AD02023)"
    BASRAH_LIGHT_AD02024="BASRAH LIGHT (AD02024)"
    BASRAH_MEDIUM_AD02025="BASRAH MEDIUM (AD02025)"
    BCF_24_AD02026="BCF 24 (AD02026)"
    BEATRICE_AD02027="BEATRICE (AD02027)"
    BEKAPAI_AD02028="BEKAPAI (AD02028)"
    BEKOK_AD02029="BEKOK (AD02029)"
    BELAYIM_AD02030="BELAYIM (AD02030)"
    BELRIDGE_HEAVY_AD02032="BELRIDGE HEAVY (AD02032)"
    BENT_HORN_AD02033="BENT HORN (AD02033)"
    BERRI_AD02035="BERRI (AD02035)"
    BETA_AD02037="BETA (AD02037)"
    BOMBAY_HIGH_AD02039="BOMBAY HIGH (AD02039)"
    BONNY_LIGHT_AD02040="BONNY LIGHT (AD02040)"
    BONNY_MEDIUM_AD02041="BONNY MEDIUM (AD02041)"
    BOSCAN_AD02042="BOSCAN (AD02042)"
    BOW_RIVER_HEAVY_AD02044="BOW RIVER HEAVY (AD02044)"
    BRAE_AD02045="BRAE (AD02045)"
    BRASS_RIVER_AD02046="BRASS RIVER (AD02046)"
    BREGA_AD02047="BREGA (AD02047)"
    BUNKER_C_FUEL_OIL_AD02051="BUNKER C FUEL OIL (AD02051)"
    BUNKER_C_FUEL_OIL_ALASKA_AD02052="BUNKER C FUEL OIL (ALASKA) (AD02052)"
    BUNK_FUEL_OIL_IRVING_WHALE_AD02053="BUNK FUEL OIL (IRVING WHALE) (AD02053)"
    BUNYU_AD02054="BUNYU (AD02054)"
    BURGAN_AD02055="BURGAN (AD02055)"
    CABINDA_AD02056="CABINDA (AD02056)"
    CALIFORNIA_API_11_AD02057="CALIFORNIA (API 11) (AD02057)"
    CALIFORNIA_API_15_AD02058="CALIFORNIA (API 15) (AD02058)"
    CANADON_SECO_AD02059="CANADON SECO (AD02059)"
    CANO_LIMON_AD02060="CANO LIMON (AD02060)"
    CARPINTERIA_AD02061="CARPINTERIA (AD02061)"
    CATALYTIC_CRACKING_FEED_AD02063="CATALYTIC CRACKING FEED (AD02063)"
    CEUTA_AD02064="CEUTA (AD02064)"
    CHAMPION_EXPORT_AD02065="CHAMPION EXPORT (AD02065)"
    CINTA_AD02066="CINTA (AD02066)"
    COAL_OIL_POINT_SEEP_OIL_AD02067="COAL OIL POINT SEEP OIL (AD02067)"
    COHASSET_AD02068="COHASSET (AD02068)"
    COLD_LAKE_BITUMEN_AD02069="COLD LAKE BITUMEN (AD02069)"
    COLD_LAKE_BLEND_AD02070="COLD LAKE BLEND (AD02070)"
    COOPER_BASIN_AD02072="COOPER BASIN (AD02072)"
    CORMORANT_NORTH_AD02073="CORMORANT NORTH (AD02073)"
    CORMORANT_SOUTH_AD02074="CORMORANT SOUTH (AD02074)"
    COSSACK_AD02076="COSSACK (AD02076)"
    CUSIANA_AD02077="CUSIANA (AD02077)"
    DAN_AD02079="DAN (AD02079)"
    DANISH_NORTH_SEA_AD02080="DANISH NORTH SEA (AD02080)"
    DIESEL_FUEL_OIL_ALASKA_AD02081="DIESEL FUEL OIL (ALASKA) (AD02081)"
    DIESEL_FUEL_OIL_SOUTHERN_USA_1994_AD02083="DIESEL FUEL OIL (SOUTHERN USA 1994) (AD02083)"
    DIESEL_FUEL_OIL_SOUTHERN_USA_1997_AD02084="DIESEL FUEL OIL (SOUTHERN USA 1997) (AD02084)"
    DJENO_BLEND_AD02086="DJENO BLEND (AD02086)"
    DORROOD_AD02087="DORROOD (AD02087)"
    DOS_CUADRAS_AD02088="DOS CUADRAS (AD02088)"
    DUBAI_AD02089="DUBAI (AD02089)"
    DUKHAN_AD02090="DUKHAN (AD02090)"
    EAST_ZEIT_MIX_AD02093="EAST ZEIT MIX (AD02093)"
    EKOFISK_AD02094="EKOFISK (AD02094)"
    ELECTRICAL_INSULATING_OIL_VOLTESSO_35_AD02098="ELECTRICAL INSULATING OIL (VOLTESSO 35) (AD02098)"
    EMERALD_AD02099="EMERALD (AD02099)"
    EMPIRE_AD02100="EMPIRE (AD02100)"
    FORCADOS_AD02101="FORCADOS (AD02101)"
    KOME_AD02102="KOME (AD02102)"
    MIANDOUM_AD02103="MIANDOUM (AD02103)"
    BOLOBO_AD02104="BOLOBO (AD02104)"
    CUSIANA_MOTIVA_ENTERPRISES_LLC_AD02105="CUSIANA, MOTIVA ENTERPRISES LLC (AD02105)"
    RABI_MOTIVA_ENTERPRISES_LLC_AD02107="RABI, MOTIVA ENTERPRISES LLC (AD02107)"
    NKOSSA_EXP_BLEND_CHEVRON_AD02108="NKOSSA EXP BLEND, CHEVRON (AD02108)"
    ANTAN_HUVENSA_AD02109="ANTAN, HUVENSA (AD02109)"
    ENDICOTT_AD02110="ENDICOTT (AD02110)"
    EOCENE_AD02111="EOCENE (AD02111)"
    ES_SIDER_AD02112="ES SIDER (AD02112)"
    ESCALANTE_AD02113="ESCALANTE (AD02113)"
    ESCRAVOS_AD02114="ESCRAVOS (AD02114)"
    ESPOIR_AD02115="ESPOIR (AD02115)"
    EUGENE_ISLAND_BLOCK_32_AD02116="EUGENE ISLAND BLOCK 32 (AD02116)"
    EUGENE_ISLAND_BLOCK_43_AD02117="EUGENE ISLAND BLOCK 43 (AD02117)"
    EVERDELL_AD02118="EVERDELL (AD02118)"
    FEDERATED_1994_AD02119="FEDERATED (1994) (AD02119)"
    FEDERATED_1998_AD02120="FEDERATED (1998) (AD02120)"
    FEDERATED_SOCSEX_AD02121="FEDERATED (SOCSEX) (AD02121)"
    FEDERATED_LIGHT_AND_MEDIUM_AD02122="FEDERATED LIGHT AND MEDIUM (AD02122)"
    FLOTTA_AD02123="FLOTTA (AD02123)"
    FLUID_CATALYTIC_CRACKER_FEED_AD02124="FLUID CATALYTIC CRACKER FEED (AD02124)"
    FLUID_CATALYTIC_CRACKER_HEAVY_CYCLE_OIL_AD02125="FLUID CATALYTIC CRACKER HEAVY CYCLE OIL (AD02125)"
    FLUID_CATALYTIC_CRACKER_LIGHT_CYCLE_OIL_AD02126="FLUID CATALYTIC CRACKER LIGHT CYCLE OIL (AD02126)"
    FLUID_CATALYTIC_CRACKER_MEDIUM_CYCLE_OIL_AD02127="FLUID CATALYTIC CRACKER MEDIUM CYCLE OIL (AD02127)"
    FORCADOS_BLEND_AD02129="FORCADOS BLEND (AD02129)"
    FOROOZAN_AD02130="FOROOZAN (AD02130)"
    FORTIES_BLEND_AD02131="FORTIES BLEND (AD02131)"
    FUEL_OIL_NO1_JP_6_AD02136="FUEL OIL NO.1 (JP-6) (AD02136)"
    FUEL_OIL_NO2_HIGH_AROMATIC_CONTENT_HEATING_OIL_AD02139="FUEL OIL NO.2 (HIGH AROMATIC CONTENT HEATING OIL) (AD02139)"
    FULMAR_AD02144="FULMAR (AD02144)"
    GALEOTA_MIX_AD02145="GALEOTA MIX (AD02145)"
    GAMBA_AD02146="GAMBA (AD02146)"
    GARDEN_BANKS_BLOCK_387_AD02147="GARDEN BANKS BLOCK 387 (AD02147)"
    GARDEN_BANKS_BLOCK_426_AD02148="GARDEN BANKS BLOCK 426 (AD02148)"
    GASOLINE_UNLEADED_SHELL_AD02153="GASOLINE (UNLEADED), SHELL (AD02153)"
    GENESIS_AD02156="GENESIS (AD02156)"
    GIPPSLAND_AD02157="GIPPSLAND (AD02157)"
    GORM_AD02158="GORM (AD02158)"
    GRANITE_POINT_AD02159="GRANITE POINT (AD02159)"
    GREEN_CANYON_BLOCK_109_AD02160="GREEN CANYON BLOCK 109 (AD02160)"
    GREEN_CANYON_BLOCK_184_AD02161="GREEN CANYON BLOCK 184 (AD02161)"
    GREEN_CANYON_BLOCK_65_AD02162="GREEN CANYON BLOCK 65 (AD02162)"
    GRIFFIN_AD02163="GRIFFIN (AD02163)"
    GULF_OF_SUEZ_MIX_AD02164="GULF OF SUEZ MIX (AD02164)"
    GULLFAKS_AD02165="GULLFAKS (AD02165)"
    HANDIL_AD02166="HANDIL (AD02166)"
    HEBRON_AD02168="HEBRON (AD02168)"
    HEIDRUN_AD02169="HEIDRUN (AD02169)"
    HIBERNIA_AD02170="HIBERNIA (AD02170)"
    HIBERNIA_EPA_86_AD02171="HIBERNIA (EPA 86) (AD02171)"
    HIGH_VISCOSITY_FUEL_OIL_AD02172="HIGH VISCOSITY FUEL OIL (AD02172)"
    HONDO_BLEND_AD02174="HONDO BLEND (AD02174)"
    HONDO_MONTEREY_AD02175="HONDO MONTEREY (AD02175)"
    HONDO_SANDSTONE_AD02176="HONDO SANDSTONE (AD02176)"
    HOUT_AD02177="HOUT (AD02177)"
    HYDRA_AD02178="HYDRA (AD02178)"
    IF_30_FUEL_OIL_AD02179="IF-30 FUEL OIL (AD02179)"
    IF_30_FUEL_OIL_SVALBARD_AD02180="IF-30 FUEL OIL (SVALBARD) (AD02180)"
    IF_30_FUEL_OIL_180_AD02181="IF-30 FUEL OIL 180 (AD02181)"
    INTERMEDIATE_FUEL_OIL_180_SOCSEX_AD02182="INTERMEDIATE FUEL OIL 180 (SOCSEX) (AD02182)"
    INTERMEDIATE_FUEL_OIL_300_AD02183="INTERMEDIATE FUEL OIL 300 (AD02183)"
    INTERMEDIATE_FUEL_OIL_300_SOCSEX_AD02184="INTERMEDIATE FUEL OIL 300 (SOCSEX) (AD02184)"
    IRANIAN_HEAVY_AD02186="IRANIAN HEAVY (AD02186)"
    IRANIAN_LIGHT_AD02187="IRANIAN LIGHT (AD02187)"
    ISTHMUS_AD02189="ISTHMUS (AD02189)"
    JATIBARANG_AD02192="JATIBARANG (AD02192)"
    KHALDA_AD02197="KHALDA (AD02197)"
    KIMKOL_AD02198="KIMKOL (AD02198)"
    KIRKUK_BLEND_AD02199="KIRKUK BLEND (AD02199)"
    KITTIWAKE_AD02200="KITTIWAKE (AD02200)"
    KOAKOAK_AD02201="KOAKOAK (AD02201)"
    KOLE_MARINE_BLEND_AD02202="KOLE MARINE BLEND (AD02202)"
    KOMINEFT_AD02203="KOMINEFT (AD02203)"
    KOPANOAR_AD02204="KOPANOAR (AD02204)"
    KUWAIT_AD02207="KUWAIT (AD02207)"
    LABUAN_BLEND_AD02209="LABUAN BLEND (AD02209)"
    LAGO_AD02210="LAGO (AD02210)"
    LAGO_TRECO_AD02211="LAGO TRECO (AD02211)"
    LAGOMEDIO_AD02212="LAGOMEDIO (AD02212)"
    LEONA_AD02213="LEONA (AD02213)"
    LIVERPOOL_BAY_AD02214="LIVERPOOL BAY (AD02214)"
    LLOYDMINSTER_AD02215="LLOYDMINSTER (AD02215)"
    LORETO_AD02216="LORETO (AD02216)"
    LOUISIANA_AD02217="LOUISIANA (AD02217)"
    LUBRICATING_OIL_AIR_COMPRESSOR_NEW_AD02220="LUBRICATING OIL (AIR COMPRESSOR) NEW (AD02220)"
    LUBRICATING_OIL_AIR_COMPRESSOR_USED_AD02221="LUBRICATING OIL (AIR COMPRESSOR) USED (AD02221)"
    LUCULA_AD02240="LUCULA (AD02240)"
    MAGNUS_AD02241="MAGNUS (AD02241)"
    MAIN_PASS_BLOCK_306_AD02242="MAIN PASS BLOCK 306 (AD02242)"
    MAIN_PASS_BLOCK_37_AD02243="MAIN PASS BLOCK 37 (AD02243)"
    MALONGO_AD02244="MALONGO (AD02244)"
    MANDJI_AD02245="MANDJI (AD02245)"
    MARGHAM_LIGHT_AD02246="MARGHAM LIGHT (AD02246)"
    MARINE_DIESEL_FUEL_OIL_AD02247="MARINE DIESEL FUEL OIL (AD02247)"
    MARINE_INTERMEDIATE_FUEL_OIL_AD02250="MARINE INTERMEDIATE FUEL OIL (AD02250)"
    MARS_BLEND_AD02251="MARS BLEND (AD02251)"
    MARS_TLP_AD02252="MARS TLP (AD02252)"
    MAUREEN_AD02253="MAUREEN (AD02253)"
    MAYA_AD02254="MAYA (AD02254)"
    MAYA_1997_AD02255="MAYA (1997) (AD02255)"
    MEDANITO_AD02256="MEDANITO (AD02256)"
    MEREY_AD02257="MEREY (AD02257)"
    MIRI_LIGHT_AD02259="MIRI LIGHT (AD02259)"
    MISSISSIPPI_CANYON_BLOCK_194_AD02260="MISSISSIPPI CANYON BLOCK 194 (AD02260)"
    MISSISSIPPI_CANYON_BLOCK_72_AD02261="MISSISSIPPI CANYON BLOCK 72 (AD02261)"
    MONTROSE_AD02263="MONTROSE (AD02263)"
    MOUSSE_MIX_PETAWAWA_AD02264="MOUSSE MIX (PETAWAWA) (AD02264)"
    MURBAN_AD02266="MURBAN (AD02266)"
    MURCHISON_AD02267="MURCHISON (AD02267)"
    NEPTUNE_SPAR_AD02273="NEPTUNE SPAR (AD02273)"
    NEWFOUNDLAND_OFFSHORE_BURN_EXP_SAMPLE_1_AD02275="NEWFOUNDLAND OFFSHORE BURN EXP SAMPLE #1 (AD02275)"
    NEWFOUNDLAND_OFFSHORE_BURN_EXP_SAMPLE_12_AD02276="NEWFOUNDLAND OFFSHORE BURN EXP SAMPLE #12 (AD02276)"
    NEWFOUNDLAND_OFFSHORE_BURN_EXP_SAMPLE_15_AD02277="NEWFOUNDLAND OFFSHORE BURN EXP SAMPLE #15 (AD02277)"
    NEWFOUNDLAND_OFFSHORE_BURN_EXP_SAMPLE_4_AD02278="NEWFOUNDLAND OFFSHORE BURN EXP SAMPLE #4 (AD02278)"
    NEWFOUNDLAND_OFFSHORE_BURN_EXP_SAMPLE_5_AD02279="NEWFOUNDLAND OFFSHORE BURN EXP SAMPLE #5 (AD02279)"
    NEWFOUNDLAND_OFFSHORE_BURN_EXP_SAMPLE_7_AD02280="NEWFOUNDLAND OFFSHORE BURN EXP SAMPLE #7 (AD02280)"
    NINIAN_BLEND_AD02281="NINIAN BLEND (AD02281)"
    NORMAN_WELLS_AD02282="NORMAN WELLS (AD02282)"
    POINT_ARGUELLO_COMINGLED_AD02284="POINT ARGUELLO COMINGLED (AD02284)"
    POINT_ARGUELLO_HEAVY_AD02286="POINT ARGUELLO HEAVY (AD02286)"
    OMAN_EXPORT_AD02287="OMAN EXPORT (AD02287)"
    ORIENTE_AD02289="ORIENTE (AD02289)"
    ORIMULSION_100_AD02290="ORIMULSION-100 (AD02290)"
    OSEBERG_AD02293="OSEBERG (AD02293)"
    PANUKE_AD02294="PANUKE (AD02294)"
    PITAS_POINT_AD02297="PITAS POINT (AD02297)"
    PLATFORM_GAIL_AD02298="PLATFORM GAIL (AD02298)"
    PLATFORM_HOLLY_AD02299="PLATFORM HOLLY (AD02299)"
    PLATFORM_IRENE_AD02300="PLATFORM IRENE (AD02300)"
    POINT_ARGUELLO_LIGHT_AD02301="POINT ARGUELLO LIGHT (AD02301)"
    PORT_HUENEME_AD02302="PORT HUENEME (AD02302)"
    PRUDHOE_BAY_AD02304="PRUDHOE BAY (AD02304)"
    PRUDHOE_BAY_1995_AD02305="PRUDHOE BAY (1995) (AD02305)"
    PULAI_AD02306="PULAI (AD02306)"
    QATAR_MARINE_AD02307="QATAR MARINE (AD02307)"
    QUA_IBOE_AD02308="QUA IBOE (AD02308)"
    RANGELY_AD02311="RANGELY (AD02311)"
    RINCON_DE_LOS_SAUCES_AD02312="RINCON DE LOS SAUCES (AD02312)"
    ROSTAM_AD02314="ROSTAM (AD02314)"
    SABLE_ISLAND_CONDENSATE_AD02315="SABLE ISLAND CONDENSATE (AD02315)"
    SAHARAN_BLEND_AD02316="SAHARAN BLEND (AD02316)"
    SAKHALIN_AD02317="SAKHALIN (AD02317)"
    SALADIN_AD02318="SALADIN (AD02318)"
    SALAWATI_AD02319="SALAWATI (AD02319)"
    SALMON_AD02320="SALMON (AD02320)"
    SANGA_SANGA_AD02322="SANGA SANGA (AD02322)"
    SANTA_CLARA_AD02323="SANTA CLARA (AD02323)"
    SCOTIAN_LIGHT_AD02325="SCOTIAN LIGHT (AD02325)"
    SEPINGGAN_YAKIN_MIXED_41_AD02326="SEPINGGAN-YAKIN MIXED (4:1) (AD02326)"
    SERIA_LIGHT_AD02327="SERIA LIGHT (AD02327)"
    SHENGLI_AD02329="SHENGLI (AD02329)"
    SHIP_SHOAL_BLOCK_239_AD02330="SHIP SHOAL BLOCK 239 (AD02330)"
    SHIP_SHOAL_BLOCK_269_AD02331="SHIP SHOAL BLOCK 269 (AD02331)"
    SIBERIAN_LIGHT_AD02332="SIBERIAN LIGHT (AD02332)"
    SIRRI_AD02333="SIRRI (AD02333)"
    SKUA_AD02335="SKUA (AD02335)"
    SOCKEYE_AD02336="SOCKEYE (AD02336)"
    SOCKEYE_COMINGLED_AD02337="SOCKEYE COMINGLED (AD02337)"
    SOCKEYE_SOUR_AD02338="SOCKEYE SOUR (AD02338)"
    SOCKEYE_SWEET_AD02339="SOCKEYE SWEET (AD02339)"
    SOROOSH_AD02340="SOROOSH (AD02340)"
    SOUEDIE_AD02341="SOUEDIE (AD02341)"
    SOUTH_PASS_BLOCK_60_AD02344="SOUTH PASS BLOCK 60 (AD02344)"
    SOUTH_PASS_BLOCK_67_AD02345="SOUTH PASS BLOCK 67 (AD02345)"
    SOUTH_PASS_BLOCK_93_AD02346="SOUTH PASS BLOCK 93 (AD02346)"
    SOUTH_TIMBALIER_BLOCK_130_AD02347="SOUTH TIMBALIER BLOCK 130 (AD02347)"
    SOYO_BLEND_AD02349="SOYO BLEND (AD02349)"
    STATFJORD_AD02351="STATFJORD (AD02351)"
    SUMATRAN_HEAVY_AD02352="SUMATRAN HEAVY (AD02352)"
    SUMATRAN_LIGHT_AD02353="SUMATRAN LIGHT (AD02353)"
    SWANSON_RIVER_AD02354="SWANSON RIVER (AD02354)"
    SYNTHETIC_AD02356="SYNTHETIC (AD02356)"
    TAKULA_AD02358="TAKULA (AD02358)"
    TAPIS_AD02359="TAPIS (AD02359)"
    TAPIS_BLEND_AD02360="TAPIS BLEND (AD02360)"
    TARTAN_AD02362="TARTAN (AD02362)"
    TEMBUNGO_AD02363="TEMBUNGO (AD02363)"
    TERRA_NOVA_AD02364="TERRA NOVA (AD02364)"
    TERRA_NOVA_1994_AD02365="TERRA NOVA (1994) (AD02365)"
    TERRA_NOVA_PETAWAWA_AD02366="TERRA NOVA (PETAWAWA) (AD02366)"
    TERRA_NOVA_SOCSEX_AD02367="TERRA NOVA (SOCSEX) (AD02367)"
    THEVENARD_ISLAND_AD02368="THEVENARD ISLAND (AD02368)"
    THISTLE_AD02369="THISTLE (AD02369)"
    TIA_JUANA_HEAVY_AD02370="TIA JUANA HEAVY (AD02370)"
    TIA_JUANA_LIGHT_AD02371="TIA JUANA LIGHT (AD02371)"
    TRADING_BAY_AD02373="TRADING BAY (AD02373)"
    TRANSMOUNTAIN_BLEND_AD02374="TRANSMOUNTAIN BLEND (AD02374)"
    UDANG_AD02376="UDANG (AD02376)"
    VASCONIA_AD02381="VASCONIA (AD02381)"
    VIOSCA_KNOLL_BLOCK_826_AD02382="VIOSCA KNOLL BLOCK 826 (AD02382)"
    VIOSCA_KNOLL_BLOCK_990_AD02383="VIOSCA KNOLL BLOCK 990 (AD02383)"
    WABASCA_BITUMEN_AD02384="WABASCA BITUMEN (AD02384)"
    WAINWRIGHT_KINSELLA_AD02385="WAINWRIGHT-KINSELLA (AD02385)"
    WALIO_AD02386="WALIO (AD02386)"
    WAXY_LIGHT_HEAVY_BLEND_AD02387="WAXY LIGHT HEAVY BLEND (AD02387)"
    WEST_DELTA_BLOCK_30_AD02388="WEST DELTA BLOCK 30 (AD02388)"
    WEST_SAK_AD02390="WEST SAK (AD02390)"
    WEST_TEXAS_INTERMEDIATE_AD02391="WEST TEXAS INTERMEDIATE (AD02391)"
    WEST_TEXAS_SOUR_AD02392="WEST TEXAS SOUR (AD02392)"
    ZAIRE_AD02394="ZAIRE (AD02394)"
    ZAKUM_AD02395="ZAKUM (AD02395)"
    ZARZITINE_AD02396="ZARZITINE (AD02396)"
    ZUEITINA_AD02397="ZUEITINA (AD02397)"
    ZULUFMARJAN_AD02398="ZULUF/MARJAN (AD02398)"
    GULF_ALBERTA_LIGHT_AND_MEDIUM_AD02401="GULF ALBERTA LIGHT AND MEDIUM (AD02401)"
    KOAKOAK_0_22A_AD02402="KOAKOAK 0-22A (AD02402)"
    LUCINA_MARINE_AD02403="LUCINA MARINE (AD02403)"
    BELINDA_AMSA_AD02408="BELINDA, AMSA (AD02408)"
    UPPER_ZAKUM_AMSA_AD02417="UPPER ZAKUM, AMSA (AD02417)"
    JET_FUEL_TESORO_AD02425="JET FUEL, TESORO (AD02425)"
    HOME_HEATING_OIL_AD02426="HOME HEATING OIL (AD02426)"
    IFO_300_AD02428="IFO 300 (AD02428)"
    JP_5_AD02430="JP-5 (AD02430)"
    FUEL_OIL_NO6_AD02431="FUEL OIL NO.6 (AD02431)"
    JP_8_AD02433="JP-8 (AD02433)"
    JP_8_AD02434="JP-8 (AD02434)"
    KUWAIT_AD02435="KUWAIT (AD02435)"
    DIESEL_FUEL_OIL_NO2_BONDED_TESORO_AD02436="DIESEL FUEL OIL NO.2 (BONDED), TESORO (AD02436)"
    STAR_4_EQUILON_AD02437="STAR 4, EQUILON (AD02437)"
    STAR_5_EQUILON_AD02438="STAR 5, EQUILON (AD02438)"
    STAR_12_EQUILON_AD02439="STAR 12, EQUILON (AD02439)"
    SAKHALIN_II_AD02440="SAKHALIN II (AD02440)"
    ESCALANTE_ITS_AD02441="ESCALANTE, ITS (AD02441)"
    MARINE_DIESEL_US_NAVY_AD02447="MARINE DIESEL, U.S. NAVY (AD02447)"
    LUCKENBACH_FUEL_OIL_AD02448="LUCKENBACH FUEL OIL (AD02448)"
    SCHIEHALLION_BLEND_STATOIL_AD02450="SCHIEHALLION BLEND, STATOIL (AD02450)"
    TROLL_STATOIL_AD02452="TROLL, STATOIL (AD02452)"
    GLITNE_STATOIL_AD02454="GLITNE, STATOIL (AD02454)"
    NORNE_STATOIL_AD02455="NORNE, STATOIL (AD02455)"
    LUFENG_STATOIL_AD02456="LUFENG, STATOIL (AD02456)"
    VARG_STATOIL_AD02458="VARG, STATOIL (AD02458)"
    GULLFAKS_C_STATOIL_AD02459="GULLFAKS C, STATOIL (AD02459)"
    GULLFAKS_A_STATOIL_AD02460="GULLFAKS A, STATOIL (AD02460)"
    OSEBERG_BLEND_STATOIL_AD02462="OSEBERG BLEND, STATOIL (AD02462)"
    EKOFISK_BLEND_STATOIL_AD02463="EKOFISK BLEND, STATOIL (AD02463)"
    STATFJORD_BLEND_STATOIL_AD02464="STATFJORD BLEND, STATOIL (AD02464)"
    ASGARD_STATOIL_AD02466="ASGARD, STATOIL (AD02466)"
    NJORD_STATOIL_AD02467="NJORD, STATOIL (AD02467)"
    GIRASSOL_AD02470="GIRASSOL (AD02470)"
    JOTUN_OIL__GAS_JOURNAL_AD02471="JOTUN, OIL & GAS JOURNAL (AD02471)"
    PIERCE_OIL__GAS_JOURNAL_AD02472="PIERCE, OIL & GAS JOURNAL (AD02472)"
    TEMPA_ROSSA_OIL__GAS_JOURNAL_AD02473="TEMPA ROSSA, OIL & GAS JOURNAL (AD02473)"
    ZUATA_SWEET_OIL__GAS_JOURNAL_AD02474="ZUATA SWEET, OIL & GAS JOURNAL (AD02474)"
    MILNE_POINT_AD02477="MILNE POINT (AD02477)"
    LISBURNE_AD02478="LISBURNE (AD02478)"
    ALPINE_SALES_OIL_AD02479="ALPINE SALES OIL (AD02479)"
    NORTHSTAR_AD02480="NORTHSTAR (AD02480)"
    WEST_SAK_AD02481="WEST SAK (AD02481)"
    BACHAQUERO_DELAWARE_RIVER_CITGO_AD02482="BACHAQUERO-DELAWARE RIVER, CITGO (AD02482)"
    CONDENSATE_SWEET_ENCANA_CORP_AD02483="CONDENSATE (SWEET), ENCANA CORP. (AD02483)"
    EAGLE_FORD_SHALE_AD02538="EAGLE FORD SHALE (AD02538)"
    NAPO_AD02539="NAPO (AD02539)"
    US_HIGH_SWEET_CLEARBROOK_AD02540="U.S. HIGH SWEET-CLEARBROOK (AD02540)"
    ULTRA_LOW_SULFUR_DIESEL_AD02541="ULTRA LOW SULFUR DIESEL (AD02541)"
    ALPINE_AD02542="ALPINE (AD02542)"
    HOOPS_BLEND_ExxonMobil_AD02547="HOOPS BLEND, ExxonMobil (AD02547)"
    AGBAMI_STATOIL_AD02548="AGBAMI, STATOIL (AD02548)"
    ALBA_AD02549="ALBA (AD02549)"
    ALGERIAN_CONDENSATE_STATOIL_AD02550="ALGERIAN CONDENSATE, STATOIL (AD02550)"
    ALVHEIM_BLEND_STATOIL_AD02551="ALVHEIM BLEND, STATOIL (AD02551)"
    AASGARD_BLEND_STATOIL_AD02552="AASGARD BLEND, STATOIL (AD02552)"
    AZERI_BTC_STATOIL_AD02553="AZERI BTC, STATOIL (AD02553)"
    AZERI_LIGHT_STATOIL_AD02554="AZERI LIGHT, STATOIL (AD02554)"
    CLOV_STATOIL_AD02555="CLOV, STATOIL (AD02555)"
    DALIA_STATOIL_AD02556="DALIA, STATOIL (AD02556)"
    DRAUGEN_STATOIL_AD02557="DRAUGEN, STATOIL (AD02557)"
    EKOFISK_STATOIL_AD02558="EKOFISK, STATOIL (AD02558)"
    FORTIES_STATOIL_AD02559="FORTIES, STATOIL (AD02559)"
    GIMBO_STATOIL_AD02560="GIMBO, STATOIL (AD02560)"
    GIRASSOL_STATOIL_AD02561="GIRASSOL, STATOIL (AD02561)"
    GOLIAT_BLEND_STATOIL_AD02562="GOLIAT BLEND, STATOIL (AD02562)"
    GRANE_BLEND_STATOIL_AD02563="GRANE BLEND, STATOIL (AD02563)"
    GUDRUN_BLEND_STATOIL_AD02564="GUDRUN BLEND, STATOIL (AD02564)"
    GULLFAKS_STATOIL_AD02565="GULLFAKS, STATOIL (AD02565)"
    HEIDRUN_STATOIL_AD02566="HEIDRUN, STATOIL (AD02566)"
    HIBERNIA_BLEND_STATOIL_AD02567="HIBERNIA BLEND, STATOIL (AD02567)"
    HUNGO_BLEND_STATOIL_AD02569="HUNGO BLEND, STATOIL (AD02569)"
    ALASKA_NORTH_SLOPE_BP_AD02570="ALASKA NORTH SLOPE, BP (AD02570)"
    ARABIAN_LIGHT_AD02572="ARABIAN LIGHT (AD02572)"
    ALASKA_NORTH_SLOPE_2019_AD02579="ALASKA NORTH SLOPE 2019 (AD02579)"
    AMSA_Average_Very_Low_Sulfur_Fuel_Oil_VLSFO_AD02580="AMSA Average Very Low Sulfur Fuel Oil (VLSFO) (AD02580)"
    VLSFO_O_1_AMSA_AD02582="VLSFO O-1 (AMSA) (AD02582)"
    VLSFO_O_2_AMSA_AD02583="VLSFO O-2 (AMSA) (AD02583)"
    VLSFO_O_3_AMSA_AD02584="VLSFO O-3 (AMSA) (AD02584)"
    VLSFO_O_4_AMSA_AD02585="VLSFO O-4 (AMSA) (AD02585)"
    VLSFO_O_5_AMSA_AD02586="VLSFO O-5 (AMSA) (AD02586)"
    VLSFO_O_6_AMSA_AD02587="VLSFO O-6 (AMSA) (AD02587)"
    VLSFO_O_7_AMSA_AD02588="VLSFO O-7 (AMSA) (AD02588)"
    VLSFO_O_8_AMSA_AD02589="VLSFO O-8 (AMSA) (AD02589)"
    VLSFO_O_9_AMSA_AD02590="VLSFO O-9 (AMSA) (AD02590)"
    VLSFO_O_10_AMSA_AD02591="VLSFO O-10 (AMSA) (AD02591)"
    VLSFO_IM_5_IMAROS_AD02592="VLSFO IM-5 (IMAROS) (AD02592)"
    VLSFO_IM_3_IMAROS_AD02596="VLSFO IM-3 (IMAROS) (AD02596)"
    VLSFO_IM_4_IMAROS_AD02597="VLSFO IM-4 (IMAROS) (AD02597)"
    VLSFO_IM_6_IMAROS_AD02598="VLSFO IM-6 (IMAROS) (AD02598)"
    VLSFO_IM_8_IMAROS_AD02600="VLSFO IM-8 (IMAROS) (AD02600)"
    VLSFO_IM_10_IMAROS_AD02602="VLSFO IM-10 (IMAROS) (AD02602)"
    VLSFO_IM_11_IMAROS_AD02603="VLSFO IM-11 (IMAROS) (AD02603)"
    VLSFO_IM_12_IMAROS_AD02604="VLSFO IM-12 (IMAROS) (AD02604)"
    VLSFO_IM_13_IMAROS_AD02605="VLSFO IM-13 (IMAROS) (AD02605)"
    ULSFO_Gas_Oil_SINTEF_AD02606="ULSFO Gas Oil (SINTEF) (AD02606)"
    ULSFO_Marine_Gas_Oil_SINTEF_AD02607="ULSFO Marine Gas Oil (SINTEF) (AD02607)"
    ULSFO_Rotterdam_Diesel_SINTEF_AD02608="ULSFO Rotterdam Diesel (SINTEF) (AD02608)"
    VLSFO_WRG_oil_DMZ_RMA_quality_SINTEF_AD02609="VLSFO WRG oil, DMZ-RMA quality (SINTEF) (AD02609)"
    ULSFO_Heavy_Distillate_Marine_ECA_50_SINTEF_AD02610="ULSFO Heavy Distillate Marine ECA 50 (SINTEF) (AD02610)"
    Ultra_Low_Sulphur_Fuel_Oil_ULSFO_RMA_quality_oil_SINTEF_AD02611="Ultra Low Sulphur Fuel Oil (ULSFO), RMA-quality oil (SINTEF) (AD02611)"
    Nile_Blend_AD02612="Nile Blend (AD02612)"
    Nile_Blend_AD02613="Nile Blend (AD02613)"
    Alaminos_Canyon_Block_25_EC00506="Alaminos Canyon Block 25 (EC00506)"
    Alaska_North_Slope_2002_EC00507="Alaska North Slope [2002] (EC00507)"
    Alberta_Sweet_Mixed_Blend_4_EC00511="Alberta Sweet Mixed Blend #4 (EC00511)"
    Alberta_Sweet_Mixed_Blend_5_EC00512="Alberta Sweet Mixed Blend #5 (EC00512)"
    Amauligak_EC00515="Amauligak (EC00515)"
    Anadarko_HIA_376_EC00517="Anadarko HIA-376 (EC00517)"
    Arabian_Heavy_2004_EC00519="Arabian Heavy [2004] (EC00519)"
    Arabian_Light_2002_EC00523="Arabian Light [2002] (EC00523)"
    Atkinson_EC00527="Atkinson (EC00527)"
    Bunker_C_1987_EC00539="Bunker C [1987] (EC00539)"
    Bunker_C___IFO_300_1994_EC00540="Bunker C - IFO-300 [1994] (EC00540)"
    Chayvo_EC00552="Chayvo (EC00552)"
    Cook_Inlet_2003_EC00561="Cook Inlet [2003] (EC00561)"
    Diesel_2002_EC00567="Diesel [2002] (EC00567)"
    Fuel_Oil__5_EC00586="Fuel Oil # 5 (EC00586)"
    Green_Canyon_Block_200_EC00593="Green Canyon Block 200 (EC00593)"
    Hebron_M_04_2005_EC00599="Hebron M-04 [2005] (EC00599)"
    HFO_6303_2002_EC00601="HFO 6303 [2002] (EC00601)"
    Hibernia_1999_EC00604="Hibernia [1999] (EC00604)"
    Issungnak_EC00616="Issungnak (EC00616)"
    Mars_TLP_2000_EC00638="Mars TLP [2000] (EC00638)"
    Maya_2004_EC00643="Maya [2004] (EC00643)"
    Mississippi_Canyon_Block_807_2002_EC00647="Mississippi Canyon Block 807 [2002] (EC00647)"
    Morpeth_Block_EW921_EC00648="Morpeth Block EW921 (EC00648)"
    Norman_Wells_EC00654="Norman Wells (EC00654)"
    Odoptu_EC00658="Odoptu (EC00658)"
    Petronius_Block_VK786A_EC00668="Petronius Block VK786A (EC00668)"
    Platform_Elly_EC00670="Platform Elly (EC00670)"
    Prudhoe_Bay_2004_EC00679="Prudhoe Bay [2004] (EC00679)"
    Sockeye_Sour_EC00690="Sockeye Sour (EC00690)"
    South_Louisiana_EC00696="South Louisiana (EC00696)"
    South_Louisiana_EC00698="South Louisiana (EC00698)"
    Troll_EC00721="Troll (EC00721)"
    West_Delta_Block_143_EC00734="West Delta Block 143 (EC00734)"
    West_Texas_Intermediate_2001_EC00736="West Texas Intermediate [2001] (EC00736)"
    White_Rose_2000_EC00738="White Rose [2000] (EC00738)"
    Albian_Heavy_Synthetic_EC01172="Albian Heavy Synthetic (EC01172)"
    Wabiska_Heavy_EC01346="Wabiska Heavy (EC01346)"
    Independent_Hub_EC01456="Independent Hub (EC01456)"
    Neptune_BHP_2009_EC01459="Neptune BHP [2009] (EC01459)"
    Platform_Irene_EC01464="Platform Irene (EC01464)"
    Platform_Irene_Comingled_EC01465="Platform Irene Comingled (EC01465)"
    Gail_Well_E010_EC01466="Gail Well E010 (EC01466)"
    Gail_Well_E019_EC01467="Gail Well E019 (EC01467)"
    Platform_Ellen_A038_EC01482="Platform Ellen A038 (EC01482)"
    Platform_Ellen_A040_EC01483="Platform Ellen A040 (EC01483)"
    Alaska_North_Slope_2010_EC01497="Alaska North Slope [2010] (EC01497)"
    Heritage_HE_05_EC01499="Heritage HE 05 (EC01499)"
    Heritage_HE_26_EC01500="Heritage HE 26 (EC01500)"
    Dos_Cuadros_HE_05_2011_EC01822="Dos Cuadros HE-05 [2011] (EC01822)"
    Dos_Cuadros_HE_26_2011_EC01823="Dos Cuadros HE-26 [2011] (EC01823)"
    Alaska_North_Slope_2011_EC01950="Alaska North Slope [2011] (EC01950)"
    DOBA_EC01951="DOBA (EC01951)"
    Endicott_EC01952="Endicott (EC01952)"
    Harmony_EC01953="Harmony (EC01953)"
    IFO_180_EC01955="IFO 180 (EC01955)"
    North_Star_EC01956="North Star (EC01956)"
    Rock_EC01957="Rock (EC01957)"
    Terra_Nova_2011_EC01958="Terra Nova [2011] (EC01958)"
    Bakken_EC01969="Bakken (EC01969)"
    Alaska_North_Slope_2012_EC02152="Alaska North Slope [2012] (EC02152)"
    Access_West_Blend_Winter_EC02234="Access West Blend Winter (EC02234)"
    Cold_Lake_Blend_Winter_2013_EC02235="Cold Lake Blend Winter [2013] (EC02235)"
    Cold_Lake_Blend_Summer_2014_EC02427="Cold Lake Blend Summer [2014] (EC02427)"
    Synthetic_Bitumen_Blend_EC02664="Synthetic Bitumen Blend (EC02664)"
    Sweet_Synthetic_Crude_Oil_2015b_EC02681="Sweet Synthetic Crude Oil [2015b] (EC02681)"
    Sweet_Synthetic_Crude_Oil_2015a_EC02695="Sweet Synthetic Crude Oil [2015a] (EC02695)"
    Western_Canadian_Select_EC02709="Western Canadian Select (EC02709)"
    Cold_Lake_Blend_Winter_2015_EC02712="Cold Lake Blend Winter [2015] (EC02712)"
    Alaska_North_Slope_2015_EC02713="Alaska North Slope [2015] (EC02713)"
    Bunker_C_MV_Manolis_2015_May_operation_EC02932="Bunker C MV Manolis 2015-May operation (EC02932)"
    MV_Arrow_2015_EC03048="MV Arrow [2015] (EC03048)"
    Rail_Bitumen_EC03126="Rail Bitumen (EC03126)"
    Husky_Energy_SGS_EC03288="Husky Energy SGS (EC03288)"
    Diesel_Echo_Bay_B5_Biodiesel_EC03629="Diesel Echo Bay (B5 Biodiesel) (EC03629)"
    Terra_Nova_2018_EC04016="Terra Nova [2018] (EC04016)"
    Marine_Diesel_2018_EC04026="Marine Diesel [2018] (EC04026)"
    Hibernia_2018_EC04028="Hibernia [2018] (EC04028)"
    Hebron_2018_EC04029="Hebron [2018] (EC04029)"
    White_Rose_2018_EC04030="White Rose [2018] (EC04030)"
    Aasgard_Blend_EX00001="Aasgard Blend (EX00001)"
    Alaska_North_Slope_EX00002="Alaska North Slope (EX00002)"
    Azeri_Light_EX00003="Azeri Light (EX00003)"
    Balder_Blend_EX00004="Balder Blend (EX00004)"
    Banyu_Urip_EX00005="Banyu Urip (EX00005)"
    Basrah_EX00006="Basrah (EX00006)"
    Basrah_Heavy_EX00007="Basrah Heavy (EX00007)"
    Bonga_EX00008="Bonga (EX00008)"
    Brent_Blend_EX00009="Brent Blend (EX00009)"
    CLOV_EX00010="CLOV (EX00010)"
    Cold_Lake_Blend_EX00011="Cold Lake Blend (EX00011)"
    Curlew_EX00012="Curlew (EX00012)"
    Dalia_EX00013="Dalia (EX00013)"
    Doba_Blend_EX00014="Doba Blend (EX00014)"
    Ebok_EX00015="Ebok (EX00015)"
    Ekofisk_EX00016="Ekofisk (EX00016)"
    Erha_EX00017="Erha (EX00017)"
    Forties_Blend_EX00018="Forties Blend (EX00018)"
    Gindungo_EX00019="Gindungo (EX00019)"
    Gippsland_Blend_EX00020="Gippsland Blend (EX00020)"
    Girassol_EX00021="Girassol (EX00021)"
    Gorgon_EX00022="Gorgon (EX00022)"
    Grane_EX00023="Grane (EX00023)"
    Gudrun_Blend_EX00024="Gudrun Blend (EX00024)"
    Gullfaks_Blend_EX00025="Gullfaks Blend (EX00025)"
    HOOPS_Blend_EX00026="HOOPS Blend (EX00026)"
    Hebron_EX00027="Hebron (EX00027)"
    Hibernia_Blend_EX00028="Hibernia Blend (EX00028)"
    Hungo_Blend_EX00029="Hungo Blend (EX00029)"
    Jotun_Blend_EX00030="Jotun Blend (EX00030)"
    Kearl_EX00031="Kearl (EX00031)"
    Kissanje_Blend_EX00032="Kissanje Blend (EX00032)"
    Kutubu_EX00033="Kutubu (EX00033)"
    Marib_Light_EX00034="Marib Light (EX00034)"
    Mondo_EX00035="Mondo (EX00035)"
    Mostarda_EX00036="Mostarda (EX00036)"
    Ormen_Lange_EX00037="Ormen Lange (EX00037)"
    Oseberg_Blend_EX00038="Oseberg Blend (EX00038)"
    Oso_Condensate_EX00039="Oso Condensate (EX00039)"
    Pazflor_EX00040="Pazflor (EX00040)"
    Qua_Iboe_EX00041="Qua Iboe (EX00041)"
    Sable_Island_EX00042="Sable Island (EX00042)"
    Saxi_Batuque_EX00043="Saxi Batuque (EX00043)"
    Sokol_EX00044="Sokol (EX00044)"
    Statfjord_Blend_EX00045="Statfjord Blend (EX00045)"
    Tapis_EX00046="Tapis (EX00046)"
    Terengganu_Condensate_EX00047="Terengganu Condensate (EX00047)"
    Terra_Nova_EX00048="Terra Nova (EX00048)"
    Thunder_Horse_EX00049="Thunder Horse (EX00049)"
    Triton_Blend_EX00050="Triton Blend (EX00050)"
    Troll_Blend_EX00051="Troll Blend (EX00051)"
    Upper_Zakum_EX00052="Upper Zakum (EX00052)"
    Usan_EX00053="Usan (EX00053)"
    Volve_EX00054="Volve (EX00054)"
    Woollybutt_EX00055="Woollybutt (EX00055)"
    Yoho_EX00056="Yoho (EX00056)"
    Zafiro_Blend_EX00057="Zafiro Blend (EX00057)"
    Liza_EX00058="Liza (EX00058)"
    Generic_Condensate_GN00001="Generic Condensate (GN00001)"
    Generic_Diesel_GN00002="Generic Diesel (GN00002)"
    Generic_Gasoline_GN00003="Generic Gasoline (GN00003)"
    Generic_Heavy_Crude_GN00004="Generic Heavy Crude (GN00004)"
    Generic_Jet_Fuel_GN00005="Generic Jet Fuel (GN00005)"
    Generic_Light_Crude_GN00006="Generic Light Crude (GN00006)"
    Generic_Medium_Crude_GN00007="Generic Medium Crude (GN00007)"
    DMA_ULSFO_LS00009="DMA (ULSFO) (LS00009)"
    DMB_ULSFO_LS00010="DMB (ULSFO) (LS00010)"
    ALVE_2010_NO00001="ALVE 2010 (NO00001)"
    ALVHEIM_BLEND_2009_NO00002="ALVHEIM BLEND 2009 (NO00002)"
    ALVHEIM_BOA_2009_NO00003="ALVHEIM BOA 2009 (NO00003)"
    ALVHEIM_KAMELEON_2009_NO00004="ALVHEIM KAMELEON 2009 (NO00004)"
    ALVHEIM_KNELER_2009_NO00005="ALVHEIM KNELER 2009 (NO00005)"
    AVALDSNES_2012_NO00006="AVALDSNES 2012 (NO00006)"
    BALDER_2002_NO00007="BALDER 2002 (NO00007)"
    BALDER_BLEND_2010_NO00008="BALDER BLEND 2010 (NO00008)"
    BRAGE_2013_NO00009="BRAGE 2013 (NO00009)"
    BREAM_2011_NO00010="BREAM 2011 (NO00010)"
    CAURUS_2011_NO00011="CAURUS 2011 (NO00011)"
    DRAUGEN_2008_NO00012="DRAUGEN 2008 (NO00012)"
    EKOFISK_2002_NO00013="EKOFISK 2002 (NO00013)"
    EKOFISK_BLEND_2000_NO00014="EKOFISK BLEND 2000 (NO00014)"
    EKOFISK_BLEND_2011_NO00015="EKOFISK BLEND 2011 (NO00015)"
    EKOFISK_J_2015_NO00016="EKOFISK J 2015 (NO00016)"
    ELDFISK_2002_NO00017="ELDFISK 2002 (NO00017)"
    ELDFISK_B_2015_NO00018="ELDFISK B 2015 (NO00018)"
    ELDFISK_BLEND_2011_NO00019="ELDFISK BLEND 2011 (NO00019)"
    ELDFISK_KOMPLEKS_2015_NO00020="ELDFISK KOMPLEKS 2015 (NO00020)"
    ELLI_1999_NO00021="ELLI 1999 (NO00021)"
    ELLI_SOUTH_1999_NO00022="ELLI SOUTH 1999 (NO00022)"
    EMBLA_2002_NO00023="EMBLA 2002 (NO00023)"
    FORSETI_2002_NO00024="FORSETI 2002 (NO00024)"
    FOSSEKALL_2013_NO00025="FOSSEKALL 2013 (NO00025)"
    FRAM_2013_NO00026="FRAM 2013 (NO00026)"
    FROY_1996_NO00027="FROY 1996 (NO00027)"
    GARANTIANA_2013_NO00028="GARANTIANA 2013 (NO00028)"
    GAUPE_2011_NO00029="GAUPE 2011 (NO00029)"
    GJOA_2011_NO00030="GJOA 2011 (NO00030)"
    GLITNE_2002_NO00031="GLITNE 2002 (NO00031)"
    GOLIAT_BLEND_5050_2008_NO00032="GOLIAT BLEND 50/50 2008 (NO00032)"
    GOLIAT_BLEND_7030_2008_NO00033="GOLIAT BLEND 70/30 2008 (NO00033)"
    GOLIAT_KOBBE_2008_NO00034="GOLIAT KOBBE 2008 (NO00034)"
    GOLIAT_REALGRUNNEN_2008_NO00035="GOLIAT REALGRUNNEN 2008 (NO00035)"
    GRANE_1997_NO00036="GRANE 1997 (NO00036)"
    GROSBEAK_2012_NO00037="GROSBEAK 2012 (NO00037)"
    GUDRUN_2012_NO00038="GUDRUN 2012 (NO00038)"
    GULLFAKS_A_BLEND_2010_NO00039="GULLFAKS A BLEND 2010 (NO00039)"
    GULLFAKS_C_BLEND_2010_NO00040="GULLFAKS C BLEND 2010 (NO00040)"
    GULLFAKS_SOR_1996_NO00041="GULLFAKS SOR 1996 (NO00041)"
    GYDA_2002_NO00042="GYDA 2002 (NO00042)"
    HAVIS_2013_NO00043="HAVIS 2013 (NO00043)"
    HEIDRUN_EXPORT_BLEND_2004_NO00044="HEIDRUN EXPORT BLEND 2004 (NO00044)"
    HEIDRUN_TILJE_2004_NO00045="HEIDRUN TILJE 2004 (NO00045)"
    HEIDRUN_AaRE_2004_NO00046="HEIDRUN AaRE 2004 (NO00046)"
    HULDRA_KONDENSAT_1998_NO00047="HULDRA KONDENSAT 1998 (NO00047)"
    IFO_180LS_2014_NO00048="IFO-180LS 2014 (NO00048)"
    IFO_180NS_2014_NO00049="IFO-180NS 2014 (NO00049)"
    IFO_80LS_2014_NO00050="IFO-80LS 2014 (NO00050)"
    IFO_380LS_2014_NO00051="IFO-380LS 2014 (NO00051)"
    IVAR_AASEN_2012_NO00052="IVAR AASEN 2012 (NO00052)"
    JORDBAER_2011_NO00053="JORDBAER 2011 (NO00053)"
    KRISTIN_2006_NO00054="KRISTIN 2006 (NO00054)"
    KVITEBJORN_2009_NO00055="KVITEBJORN 2009 (NO00055)"
    LAVRANS_1997_NO00056="LAVRANS 1997 (NO00056)"
    LILLEFRIGG_KONDENSAT_1996_NO00057="LILLEFRIGG KONDENSAT 1996 (NO00057)"
    LINERLE_2005_NO00058="LINERLE 2005 (NO00058)"
    LUNO_2011_NO00059="LUNO 2011 (NO00059)"
    LUNO_II_2014_NO00060="LUNO II 2014 (NO00060)"
    MARIA_2013_NO00061="MARIA 2013 (NO00061)"
    MIDGARD_1991_NO00062="MIDGARD 1991 (NO00062)"
    MORVIN_2008_NO00063="MORVIN 2008 (NO00063)"
    NJORD_1997_NO00064="NJORD 1997 (NO00064)"
    NORNE_1997_NO00065="NORNE 1997 (NO00065)"
    NORNE_BLEND_2010_NO00066="NORNE BLEND 2010 (NO00066)"
    ORMEN_LANGE_KONDENSAT_2008_NO00067="ORMEN LANGE KONDENSAT 2008 (NO00067)"
    OSEBERG_A_2013_NO00068="OSEBERG A 2013 (NO00068)"
    OSEBERG_SOR_2013_NO00069="OSEBERG SOR 2013 (NO00069)"
    OSEBERG_OST_2013_NO00070="OSEBERG OST 2013 (NO00070)"
    OSELVAR_2012_NO00071="OSELVAR 2012 (NO00071)"
    RINGHORNE_2002_NO00072="RINGHORNE 2002 (NO00072)"
    SKARFJELL_2014_NO00073="SKARFJELL 2014 (NO00073)"
    SKARV_2004_NO00074="SKARV 2004 (NO00074)"
    SKARV_KONDENSAT_2014_NO00075="SKARV KONDENSAT 2014 (NO00075)"
    SKRUGARD_2012_NO00076="SKRUGARD 2012 (NO00076)"
    SLEIPNER_KONDENSAT_2002_NO00077="SLEIPNER KONDENSAT 2002 (NO00077)"
    SLEIPNER_VEST_1998_NO00078="SLEIPNER VEST 1998 (NO00078)"
    SMORBUKK_2003_NO00079="SMORBUKK 2003 (NO00079)"
    SMORBUKK_KONDENSAT_2003_NO00080="SMORBUKK KONDENSAT 2003 (NO00080)"
    SMORBUKK_SOR_2003_NO00081="SMORBUKK SOR 2003 (NO00081)"
    SNORRE_B_2004_NO00082="SNORRE B 2004 (NO00082)"
    SNORRE_TLP_2004_NO00083="SNORRE TLP 2004 (NO00083)"
    SNOHVIT_KONDENSAT_2001_NO00084="SNOHVIT KONDENSAT 2001 (NO00084)"
    STATFJORD_A_2001_NO00085="STATFJORD A 2001 (NO00085)"
    STATFJORD_B_2001_NO00086="STATFJORD B 2001 (NO00086)"
    STATFJORD_C_2001_NO00087="STATFJORD C 2001 (NO00087)"
    STAER_2010_NO00088="STAER 2010 (NO00088)"
    TAMBAR_2002_NO00089="TAMBAR 2002 (NO00089)"
    TAU_1999_NO00090="TAU 1999 (NO00090)"
    TOR_2002_NO00091="TOR 2002 (NO00091)"
    TORDIS_2002_NO00092="TORDIS 2002 (NO00092)"
    TRESTAKK_2008_NO00093="TRESTAKK 2008 (NO00093)"
    TRYM_KONDENSAT_2011_NO00094="TRYM KONDENSAT 2011 (NO00094)"
    TYRIHANS_NORD_2004_NO00095="TYRIHANS NORD 2004 (NO00095)"
    TYRIHANS_SOR_2004_NO00096="TYRIHANS SOR 2004 (NO00096)"
    ULA_1999_NO00097="ULA 1999 (NO00097)"
    VALE_2014_NO00098="VALE 2014 (NO00098)"
    VALHALL_2002_NO00099="VALHALL 2002 (NO00099)"
    VARG_2000_NO00100="VARG 2000 (NO00100)"
    VESLEFRIKK_2012_NO00101="VESLEFRIKK 2012 (NO00101)"
    VIGDIS_2004_NO00102="VIGDIS 2004 (NO00102)"
    VILJE_2009_NO00103="VILJE 2009 (NO00103)"
    VISUND_2009_NO00104="VISUND 2009 (NO00104)"
    VOLUND_2010_NO00105="VOLUND 2010 (NO00105)"
    VOLVE_2006_NO00106="VOLVE 2006 (NO00106)"
    WISTING_2015_NO00107="WISTING 2015 (NO00107)"
    AASGARD_A_2003_NO00108="AASGARD A 2003 (NO00108)"
    SVALIN_2014_NO00109="SVALIN 2014 (NO00109)"
    ALTA_2016_NO00110="ALTA 2016 (NO00110)"
    DRIVIS_2017_NO00111="DRIVIS 2017 (NO00111)"
    MARTIN_LINGE_CRUDE_2016_NO00112="MARTIN LINGE CRUDE 2016 (NO00112)"
    MARTIN_LINGE_CONDENSATE_2016_NO00113="MARTIN LINGE CONDENSATE 2016 (NO00113)"
    BRYNHILD_CRUDE_2015_NO00114="BRYNHILD CRUDE 2015 (NO00114)"
    BOYLA_CRUDE_2016_NO00115="BOYLA CRUDE 2016 (NO00115)"
    WISTING_CENTRAL_2017_NO00116="WISTING CENTRAL 2017 (NO00116)"
    SIGYN_CONDENSATE_2017_NO00117="SIGYN CONDENSATE 2017 (NO00117)"
    NORNE_CRUDE_2017_NO00118="NORNE CRUDE 2017 (NO00118)"
    MARINE_GAS_OIL_500_ppm_S_2017_NO00119="MARINE GAS OIL 500 ppm S 2017 (NO00119)"
    ULTRA_LOW_SULFUR_FUEL_OIL_2017_NO00120="ULTRA LOW SULFUR FUEL OIL 2017 (NO00120)"
    HEAVY_DISTILLATE_MARINE_ECA_50_2017_NO00121="HEAVY DISTILLATE MARINE ECA 50 2017 (NO00121)"
    ROTTERDAM_DIESEL_2017_NO00122="ROTTERDAM DIESEL 2017 (NO00122)"
    GAS_OIL_10_ppm_S_2017_NO00123="GAS OIL 10 ppm S 2017 (NO00123)"
    WIDE_RANGE_GAS_OIL_2017_NO00124="WIDE RANGE GAS OIL 2017 (NO00124)"
    OSEBERG_BLEND_2007_NO00125="OSEBERG BLEND 2007 (NO00125)"
    BRASSE_2018_NO00126="BRASSE 2018 (NO00126)"
    OSEBERG_C_2013_NO00127="OSEBERG C 2013 (NO00127)"
    VEGA_CONDENSATE_2015_NO00128="VEGA CONDENSATE 2015 (NO00128)"
    FENJA_PIL_2015_NO00129="FENJA (PIL) 2015 (NO00129)"
    MARULK_2014_NO00130="MARULK 2014 (NO00130)"
    GINA_KROG_CRUDE_2018_NO00131="GINA KROG CRUDE 2018 (NO00131)"
    GUDRUN_2019_NO00132="GUDRUN 2019 (NO00132)"
    ATLA_KONDENSAT_2013_NO00133="ATLA KONDENSAT 2013 (NO00133)"
    OSEBERG_SOR_2000_NO00134="OSEBERG SOR 2000 (NO00134)"
    KVITEBJORN_2019_NO00135="KVITEBJORN 2019 (NO00135)"
    ODA_2019_NO00136="ODA 2019 (NO00136)"
    IRIS_CONDENSATE_2020_NO00137="IRIS CONDENSATE 2020 (NO00137)"
    AASTA_HANSTEEN_BLEND_2020_NO00138="AASTA HANSTEEN BLEND 2020 (NO00138)"
    VISUND_SOR_CONDENSATE_2020_NO00139="VISUND SOR CONDENSATE 2020 (NO00139)"
    VISUND_CRUDE_OIL_2020_NO00140="VISUND CRUDE OIL 2020 (NO00140)"
    OSEBERG_OST_1998_NO00141="OSEBERG OST 1998 (NO00141)"
    FROSK_2020_NO00142="FROSK 2020 (NO00142)"
    DVALIN_2020_NO00143="DVALIN 2020 (NO00143)"
    SKOGUL_2020_NO00144="SKOGUL 2020 (NO00144)"
    FOGELBERG_CONDENSATE_2021_NO00145="FOGELBERG CONDENSATE 2021 (NO00145)"
    UTGARD_CONDENSATE_2021_NO00146="UTGARD CONDENSATE 2021 (NO00146)"
    NJORD_2003_NO00147="NJORD 2003 (NO00147)"
    DUVA_2021_NO00148="DUVA 2021 (NO00148)"
    VALHALL_2021_NO00149="VALHALL 2021 (NO00149)"
    SF_NORD_BRENT_2021_NO00150="SF NORD BRENT 2021 (NO00150)"
    SYGNA_BRENT_2021_NO00151="SYGNA BRENT 2021 (NO00151)"
    ALVE_2014_NO00152="ALVE 2014 (NO00152)"
    OSEBERG_C_1995_NO00153="OSEBERG C 1995 (NO00153)"
    DUGONG_2022_NO00154="DUGONG 2022 (NO00154)"
    LILLE_PRINSEN_2022_NO00155="LILLE PRINSEN 2022 (NO00155)"
    TOR_II_2022_NO00156="TOR II 2022 (NO00156)"
    VALE_2001_NO00157="VALE 2001 (NO00157)"
    GOLIAT_REALGRUNNEN_2001_NO00158="GOLIAT REALGRUNNEN 2001 (NO00158)"
    SVALE_2010_NO00159="SVALE 2010 (NO00159)"
    BREIDABLIKK_2023_NO00160="BREIDABLIKK 2023 (NO00160)"
    YME_2023_NO00162="YME 2023 (NO00162)"
    OFELIA_2023_NO00163="OFELIA 2023 (NO00163)"
    HEIDRUN_AARE_2023_NO00164="HEIDRUN AARE 2023 (NO00164)"
    LANGFJELLET_2023_NO00165="LANGFJELLET 2023 (NO00165)"
    CALYPSO_2024_NO00166="CALYPSO 2024 (NO00166)"
    GENERIC_LIGHT_CRUDE_AD04000="GENERIC LIGHT CRUDE (AD04000)"
    GENERIC_MEDIUM_CRUDE_AD04001="GENERIC MEDIUM CRUDE (AD04001)"
    GENERIC_HEAVY_CRUDE_AD04002="GENERIC HEAVY CRUDE (AD04002)"
    GENERIC_GASOLINE_AD04003="GENERIC GASOLINE (AD04003)"
    GENERIC_FUEL_OIL_No2_AD04006="GENERIC FUEL OIL No.2 (AD04006)"
    GENERIC_DIESEL_AD04007="GENERIC DIESEL (AD04007)"
    GENERIC_HOME_HEATING_OIL_AD04008="GENERIC HOME HEATING OIL (AD04008)"
    GENERIC_INTERMEDIATE_FUEL_OIL_180_AD04009="GENERIC INTERMEDIATE FUEL OIL 180 (AD04009)"
    GENERIC_INTERMEDIATE_FUEL_OIL_300_AD04010="GENERIC INTERMEDIATE FUEL OIL 300 (AD04010)"
    GENERIC_FUEL_OIL_No_6_AD04011="GENERIC FUEL OIL No. 6 (AD04011)"
    GENERIC_BUNKER_C_AD04012="GENERIC BUNKER C (AD04012)"
    GENERIC_HEAVY_FUEL_OIL_AD04013="GENERIC HEAVY FUEL OIL (AD04013)"


class DropletSizeDistributionEnum(str, Enum):
    uniform = "uniform"
    normal = "normal"
    lognormal = "lognormal"


class OpenOilModelConfig(OceanDriftModelConfig):
    drift_model: DriftModelEnum = DriftModelEnum.OpenOil.value
    
    oil_type: OilTypeEnum = Field(
        default=OilTypeEnum.GENERIC_BUNKER_C_AD04012.value,
        description="Oil type to be used for the simulation, from the NOAA ADIOS database.",
        title="Oil Type",
        json_schema_extra={"od_mapping": "seed:oil_type", "ptm_level": 1},
    )
    
    m3_per_hour: float = Field(
        default=1,
        description="The amount (volume) of oil released per hour (or total amount if release is instantaneous)",
        title="M3 Per Hour",
        gt=0,
        json_schema_extra={"units": "m3 per hour", "od_mapping": "seed:m3_per_hour", "ptm_level": 1},
    )
    
    oil_film_thickness: float = Field(
        default=0.001,
        description="Seeding value of oil_film_thickness",
        title="Oil Film Thickness",
        json_schema_extra={
            "units": "m",
            "od_mapping": "seed:oil_film_thickness",
            "ptm_level": 3,
        },
    )
    
    droplet_size_distribution: DropletSizeDistributionEnum = Field(
        default=DropletSizeDistributionEnum.uniform.value,
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
        description="The minimum diameter of oil droplet for a subsea release, used in unifrom distribution.",
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
        description="Surface oil is emulsified, i.e. water droplets are mixed into oil due to wave mixing, with resulting increas of viscosity.",
        title="Emulsification",
        json_schema_extra={
            "od_mapping": "processes:emulsification",
            "ptm_level": 2,
        },
    )
    
    dispersion: bool = Field(
        default=True,
        description="Oil is removed from simulation (dispersed), if entrained as very small droplets.",
        title="Dispersion",
        json_schema_extra={
            "od_mapping": "processes:dispersion",
            "ptm_level": 2,
        },
    )
    
    evaporation: bool = Field(
        default=True,
        description="Surface oil is evaporated.",
        title="Evaporation",
        json_schema_extra={
            "od_mapping": "processes:evaporation",
            "ptm_level": 2,
        },
        )


    update_oilfilm_thickness: bool = Field(
        default=False,
        description="Oil film thickness is calculated at each time step. The alternative is that oil film thickness is kept constant with value provided at seeding.",
        title="Update Oilfilm Thickness",
        json_schema_extra={
            "od_mapping": "processes:update_oilfilm_thickness",
            "ptm_level": 2,
        },
        )
    
    biodegradation: bool = Field(
        default=False,
        description="Oil mass is biodegraded (eaten by bacteria).",
        title="Biodegradation",
        json_schema_extra={
            "od_mapping": "processes:biodegradation",
            "ptm_level": 2,
        },
        )
    
    # overwrite the defaults from OceanDriftModelConfig for a few inherited parameters,
    # but don't want to have to repeat the full definition
    current_uncertainty: float = FieldInfo.merge_field_infos(OceanDriftModelConfig.model_fields['current_uncertainty'],
                                                             Field(default=0.05))
    wind_uncertainty: float = FieldInfo.merge_field_infos(OceanDriftModelConfig.model_fields['wind_uncertainty'],
                                                          Field(default=0.5))
    wind_drift_factor: float = FieldInfo.merge_field_infos(OceanDriftModelConfig.model_fields['wind_drift_factor'],
                                                          Field(default=0.03))
    # OpenDrift's default is for vertical_mixing to be True but that conflicts with do3D default of False
    # vertical_mixing: bool = FieldInfo.merge_field_infos(OceanDriftModelConfig.model_fields['vertical_mixing'],
    #                                                     Field(default=True))

    
    @property
    def oil_type_input(self) -> str | None:
        """Save oil type input with both name and id"""
        if self.drift_model == "OpenOil":
            return self.oil_type
        return None
    
    @model_validator(mode='after')
    def clean_oil_type_string(self) -> Self:
        """remove id from oil_type string if needed"""
        if self.drift_model == "OpenOil":
            # only keep first part of string, which is the name of the oil
            self.oil_type = self.oil_type_input.split(" (")[0]
        return self



class LarvalFishModelConfig(OceanDriftModelConfig):
    drift_model: DriftModelEnum = DriftModelEnum.LarvalFish.value

    diameter: float = Field(
        default=0.0014,
        description="Seeding value of diameter",
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
        description="Seeding value of neutral_buoyancy_salinity",
        title="Neutral Buoyancy Salinity",
        gt=0,
        json_schema_extra= {
            "units": "PSU",
            "od_mapping": "seed:neutral_buoyancy_salinity",
            "ptm_level": 2,
        },
                                             )

    
    stage_fraction: float = Field(
        default=0.0,
        description="Seeding value of stage_fraction",
        title="Stage Fraction",
        json_schema_extra={
            "units": "",
            "od_mapping": "seed:stage_fraction",
            "ptm_level": 2,
        }
        )
    
    hatched: int = Field(
        default=0,
        description="Seeding value of hatched",
        title="Hatched",
        ge=0,
        le=1,
        json_schema_extra={
            "units": "",
            "od_mapping": "seed:hatched",
            "ptm_level": 2,
        },
    )

    length: float = Field(
        default=0,
        description="Seeding value of length",
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
        description="Seeding value of weight",
        title="Weight",
        gt=0,
        json_schema_extra={
            "units": "mg",
            "od_mapping": "seed:weight",
            "ptm_level": 2,
        },
    )
    
    # override inherited parameter defaults
    vertical_mixing: bool = FieldInfo.merge_field_infos(OceanDriftModelConfig.model_fields['vertical_mixing'],
                                                        Field(default=True))
    do3D: bool = FieldInfo.merge_field_infos(TheManagerConfig.model_fields['do3D'],
                                                        Field(default=True))


    @model_validator(mode='after')
    def check_do3D(self) -> Self:
        if not self.do3D:
            raise ValueError("do3D must be True with the LarvalFish drift model.")
            
        return self

    @model_validator(mode='after')
    def check_vertical_mixing(self) -> Self:
        if not self.vertical_mixing:
            raise ValueError("vertical_mixing must be True with the LarvalFish drift model.")
            
        return self


open_drift_mapper = {
    "OceanDrift": OceanDriftModelConfig,
    "OpenOil": OpenOilModelConfig,
    "LarvalFish": LarvalFishModelConfig,
    "Leeway": LeewayModelConfig,
}

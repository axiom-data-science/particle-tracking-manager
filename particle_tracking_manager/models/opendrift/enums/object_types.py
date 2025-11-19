"""OpenDrift object types enum definition.

How to set up the enum:
import numpy as np
import particle_tracking_manager as ptm
m = ptm.OpenDriftModel(drift_model="Leeway", steps=1)
m.setup_for_simulation()
objects = m.o.get_configspec("seed:object_type")["seed:object_type"]["enum"]
nums = np.arange(0, len(objects))
for i, s in zip(nums, objects):
    print(f'    O{i+1} = "{s}"')
"""

from enum import Enum


class ObjectTypeEnum(str, Enum):
    """Enum for object types used in OpenDrift.

    From https://github.com/OpenDrift/opendrift/blob/master/opendrift/models/OBJECTPROP.DAT
    """

    O1 = "Person-in-water (PIW), unknown state (mean values)"
    O2 = ">PIW, vertical PFD type III conscious"
    O3 = ">PIW, sitting, PFD type I or II"
    O4 = ">PIW, survival suit (face up)"
    O5 = ">PIW, scuba suit (face up)"
    O6 = ">PIW, deceased (face down)"
    O7 = "Life raft, deep ballast (DB) system, general, unknown capacity and loading (mean values)"
    O8 = ">4-14 person capacity, deep ballast system, canopy (average)"
    O9 = ">>4-14 person capacity, deep ballast system, no drogue"
    O10 = (
        ">>>4-14 person capacity, deep ballast system, canopy, no drogue, light loading"
    )
    O11 = ">>>4-14 person capacity, deep ballast system, no drogue, heavy loading"
    O12 = ">>4-14 person capacity, deep ballast system, canopy, with drogue (average)"
    O13 = ">>>4-14 person capacity, deep ballast system, canopy, with drogue, light loading"
    O14 = ">>>4-14 person capacity, deep ballast system, canopy, with drogue, heavy loading"
    O15 = ">15-50 person capacity, deep ballast system, canopy, general (mean values)"
    O16 = (
        ">>15-50 person capacity, deep ballast system, canopy, no drogue, light loading"
    )
    O17 = ">>15-50 person capacity, deep ballast system, canopy, with drogue, heavy loading"
    O18 = "Deep ballast system, general (mean values), capsized"
    O19 = "Deep ballast system, general (mean values), swamped"
    O20 = "Life-raft, shallow ballast (SB) system AND canopy, general (mean values)"
    O21 = ">Life-raft, shallow ballast system, canopy, no drogue"
    O22 = ">Life-raft, shallow ballast system AND canopy, with drogue"
    O23 = "Life-raft, shallow ballast system AND canopy, capsized"
    O24 = "Life Raft - Shallow ballast, canopy, Navy Sub Escape (SEIE) 1-man raft, NO drogue"
    O25 = "Life Raft - Shallow ballast, canopy, Navy Sub Escape (SEIE) 1-man raft, with drogue"
    O26 = "Life-raft, no ballast (NB) system, general (mean values)"
    O27 = ">Life-raft, no ballast system, no canopy, no drogue"
    O28 = ">Life-raft, no ballast system, no canopy, with drogue"
    O29 = ">Life-raft, no ballast system, with canopy, no drogue"
    O30 = ">Life-raft, no ballast system, with canopy, with drogue"
    O31 = "Survival Craft - USCG Sea Rescue Kit - 3 ballasted life rafts and 300 meter of line"
    O32 = "Life-raft, 4-6 person capacity, no ballast, with canopy, no drogue"
    O33 = "Evacuation slide with life-raft, 46 person capacity"
    O34 = "Survival Craft - SOLAS Hard Shell Life Capsule, 22 man"
    O35 = "Survival Craft - Ovatek Hard Shell Life Raft, 4 and 7-man, lightly loaded, no drogue (average)"
    O36 = ">Survival Craft - Ovatek Hard Shell Life Raft, 4 man, lightly loaded, no drogue"
    O37 = ">Survival Craft - Ovatek Hard Shell Life Raft, 7 man, lightly loaded, no drogue"
    O38 = "Survival Craft - Ovatek Hard Shell Life Raft, 4 and 7-man, fully loaded, drogued (average)"
    O39 = ">Survival Craft - Ovatek Hard Shell Life Raft, 4 man, fully loaded, drogued"
    O40 = ">Survival Craft - Ovatek Hard Shell Life Raft, 7 man, fully loaded, drogued"
    O41 = "Sea Kayak with person on aft deck"
    O42 = "Surf board with person"
    O43 = "Windsurfer with mast and sail in water"
    O44 = "Skiff - modified-v, cathedral-hull, runabout outboard powerboat"
    O45 = "Skiff, V-hull"
    O46 = "Skiffs, swamped and capsized"
    O47 = "Skiff - v-hull bow to stern (aluminum, Norway)"
    O48 = "Sport boat, no canvas (*1), modified V-hull"
    O49 = "Sport fisher, center console (*2), open cockpit"
    O50 = "Fishing vessel, general (mean values)"
    O51 = "Fishing vessel, Hawaiian Sampan (*3)"
    O52 = ">Fishing vessel, Japanese side-stern trawler"
    O53 = ">Fishing vessel, Japanese Longliner (*3)"
    O54 = ">Fishing vessel, Korean fishing vessel (*4)"
    O55 = ">Fishing vessel, Gill-netter with rear reel (*3)"
    O56 = "Coastal freighter. (*5)"
    O57 = "Sailboat Mono-hull (Average)"
    O58 = ">Sailboat Mono-hull (Dismasted, Average)"
    O59 = ">>Sailboat Mono-hull (Dismasted - rudder amidships)"
    O60 = ">>Sailboat Mono-hull (Dismasted - rudder missing)"
    O61 = ">Sailboat Mono-hull (Bare-masted,  Average)"
    O62 = ">>Sailboat Mono-hull (Bare-masted, rudder amidships)"
    O63 = ">>Sailboat Mono-hull (Bare-masted, rudder hove-to)"
    O64 = "Sailboat Mono-hull, fin keel, shallow draft (was SAILBOAT-2)"
    O65 = "Sunfish sailing dingy  -  Bare-masted, rudder missing"
    O66 = "Fishing vessel debris"
    O67 = "Self-locating datum marker buoy - no windage"
    O68 = "Navy Submarine EPIRB (SEPIRB)"
    O69 = "Bait/wharf box, holds a cubic metre of ice, mean values (*6)"
    O70 = "Bait/wharf box, holds a cubic metre of ice, lightly loaded"
    O71 = ">Bait/wharf box, holds a cubic metre of ice, full loaded"
    O72 = "55-gallon (220 l) Oil Drum"
    O73 = "Scaled down (1:3) 40-ft Container (70% submerged)"
    O74 = "20-ft Container (80% submerged)"
    O75 = "WWII L-MK2 mine"
    O76 = "Immigration vessel, Cuban refugee-raft, no sail (*7)"
    O77 = "Immigration vessel, Cuban refugee-raft, with sail (*7)"
    O78 = "Sewage floatables, tampon applicator"
    O79 = "Medical waste (mean values)"
    O80 = ">Medical waste, vials"
    O81 = ">>Medical waste, vials, large"
    O82 = ">>Medical waste, vials, small"
    O83 = ">Medical waste, syringes"
    O84 = ">>Medical waste, syringes, large"
    O85 = ">>Medical waste, syringes, small"

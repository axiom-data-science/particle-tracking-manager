# Drift Model Details and Species Reference

## Phytoplankton (Harmful Algal Blooms)

With the `Phytoplankton` model, users can transport an existing bloom forward in time or run backward in time to determine where it originated.

The model uses the OpenDrift `LarvalFish` model internally, configured for phytoplankton particle tracking. It supports optional vertical behavior modes that allow particles to actively seek preferred depths or perform diel vertical migration. There is no direct species selection parameter; instead, users configure vertical behavior and other parameters to approximate the species of interest (see {ref}`phyto_species_reference` below).

### Vertical Behavior

The `Phytoplankton` model supports active vertical positioning through the `vertical_behavior_mode` parameter:

* **`depth`**: Particles maintain a preferred depth band around `z_pref`. Useful for species that stay near the surface or at a fixed depth in the water column.
* **`dvm`** (diel vertical migration): Particles move between `z_day` (daytime depth) and `z_night` (nighttime depth). Useful for motile dinoflagellates that migrate vertically over the day-night cycle.

The speed at which particles swim toward their target depth is controlled by `w_active` (m/s).

(phyto_species_reference)=
### Phytoplankton Species Reference

When using the Phytoplankton scenario, note that it may be important to choose depths in spatial relation to the mixed layer depth (another parameter). Simplified representations of species of interest are suggested here as a reference:

**Alexandrium spp** (motile dinoflagellate):
* Vertical Speed (`w_active`): 0.0005–0.002 m/s
* Vertical Behavior Mode: `dvm` (Diel Vertical Migration)
* Daytime Depth (`z_day`): -10 to -30 meters
* Nighttime Depth (`z_night`): -1 to -5 meters

**Pseudo-nitzschia spp** (weakly/non-motile diatom):
* Vertical Speed (`w_active`): 0.0001–0.0005 m/s
* Vertical Behavior Mode: `depth` (Preferred Depth)
* Preferred Depth (`z_pref`): -0.5 to -15 meters

**Dinophysis spp** (motile dinoflagellate):
* Vertical Speed (`w_active`): 0.0005–0.002 m/s
* Vertical Behavior Mode: `dvm` (Diel Vertical Migration)
* Daytime Depth (`z_day`): -10 to -40 meters
* Nighttime Depth (`z_night`): -1 to -10 meters

```{note}
These are rough estimates and may not be accurate for all species or conditions. We recommend consulting the literature for more specific information about the species you are interested in modeling.
```


## Larval Fish

With the `LarvalFish` model, users simulate the transport and behavior of fish eggs and larvae. The user should first choose whether to initialize the simulation particles as eggs or larvae.

**Eggs** are modeled as passive particles that drift with the water until they hatch. Hatching occurs after `hatch_time_days` has passed (using `hatching_method="fixed_time"`). After hatching, particles are modeled as larvae with vertical behavior and vertical speed.

**Larvae** are initialized directly by setting `hatched=1` and immediately exhibit vertical behavior.

### Vertical Behavior

After hatching (or when initialized as larvae), larvae can actively position themselves in the water column through the `vertical_behavior_mode` parameter:

* **`depth`**: Larvae maintain a preferred depth band around `z_pref`. Useful for species that remain at a relatively fixed depth.
* **`dvm`** (diel vertical migration): Larvae move between `z_day` (daytime depth) and `z_night` (nighttime depth). Useful for species that migrate vertically over the day-night cycle.

The speed at which larvae swim toward their target depth is controlled by `w_active` (m/s).

(larval_species_reference)=
### Larval Fish Species Reference

When using the LarvalFish scenario, note that it may be important to choose depths in spatial relation to the mixed layer depth (another parameter). Simplified representations of species of interest are suggested here as a reference:

**Walleye Pollock:**
* Start as eggs with 18–25 days to hatch (`hatch_time_days`)
* Vertical Speed (`w_active`): 0.001–0.003 m/s
* Vertical Behavior Mode: `dvm` (Diel Vertical Migration)
* Daytime Depth (`z_day`): -20 to -40 meters
* Nighttime Depth (`z_night`): -5 to -20 meters

**Pacific Cod:**
* Start as eggs with 14–18 days to hatch (`hatch_time_days`)
* Vertical Speed (`w_active`): 0.001–0.003 m/s
* Vertical Behavior Mode: `dvm` (Diel Vertical Migration)
* Daytime Depth (`z_day`): -15 to -35 meters
* Nighttime Depth (`z_night`): -5 to -15 meters

**Pacific Herring:**
* Start as larvae (`hatched=1`)
* Vertical Speed (`w_active`): 0.001–0.005 m/s
* Vertical Behavior Mode: `depth` (Preferred Depth)
* Preferred Depth (`z_pref`): -5 to -15 meters

**Rockfish:**
* Start as larvae (`hatched=1`)
* Vertical Speed (`w_active`): 0.001–0.01 m/s
* Vertical Behavior Mode: `depth` (Preferred Depth)
* Preferred Depth (`z_pref`): -10 to -40 meters

**Razor Clams:**
* Start as larvae (`hatched=1`)
* Vertical Speed (`w_active`): 0.0005–0.003 m/s
* Vertical Behavior Mode: `depth` (Preferred Depth)
* Preferred Depth (`z_pref`): -3 to -10 meters

```{note}
These are rough estimates and may not be accurate for all species or conditions. We recommend consulting the literature for more specific information about the species you are interested in modeling.
```

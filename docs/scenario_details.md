# Details about individual drift models

## Harmful Algal Blooms

With the `HarmfulAlgalBloom` model, users can transport an existing bloom forward in time or run backward in time to determine where it originated.

Currently users can model *Pseudo nitzschia* as a near-surface-limited bloom that experiences high mortality when outside viable temperature and salinity ranges. No growth is currently included. This is the start to tracking relative biomass of the particles making up the bloom.

### Biomass and Mortality Framework

The model includes a simple habitat-dependent survival model in which temperature and salinity determine mortality, and biomass evolves according to an exponential decay equation.

#### Environmental Zone Classification

For each particle, the model first evaluates its local temperature and salinity to determine whether environmental conditions fall within a **preferred**, **marginal**, or **lethal** range. This classification is handled by `classify_zone`, which assigns each element to one of:

- `baseline_mortality` (preferred habitat)
- `medium_mortality` (suboptimal but viable habitat)
- `high_mortality` (outside survival limits)

Temperature classification uses four thresholds:

- `temperature_pref_min`
- `temperature_pref_max`
- `temperature_death_min`
- `temperature_death_max`

Salinity classification uses an analogous set:

- `salinity_pref_min`
- `salinity_pref_max`
- `salinity_death_min`
- `salinity_death_max`


The result is a per-particle assessment of environmental stress.

#### Mortality Rate Selection

The model then assigns each particle a mortality rate according to a tiered decision rule implemented in `choose_mortality_rate`:

1. If **temperature OR salinity** is in a `high_mortality` zone →
   use `mortality_rate_high`.

2. Else, if **either** variable is in a `medium_mortality` zone
   (and neither is high) →
   use `mortality_rate_medium`.

3. Only when **both** temperature and salinity lie in preferred ranges →
   use `mortality_rate_baseline`.

This "worst-condition wins" approach prevents double-counting stress while ensuring that the most limiting factor controls mortality.

The output is an array `mortality_rates` (units: days$^{-1}$).

#### Biomass Evolution

Biomass is updated using an exponential decay formulation. Growth is not yet implemented, so the net rate is purely negative:


This corresponds to integrating:

$$
\frac{dB}{dt} = (\text{growth_rate} - \text{mortality_rate}) \times B,
$$

so biomass decays on a timescale determined by the assigned mortality rate. Particles in lethal zones decay fastest; particles in preferred zones decay slowly (not yet implemented). The solution to the equation is

$$
B(t) = B(0) \exp^{(\text{growth_rate} - \text{mortality_rate})t}
$$

#### Deactivation of Dead Particles

Once biomass is updated, particles whose biomass falls below the threshold `biomass_dead_threshold` are removed from the simulation (or deactivated). Deactivated particles no longer participate in advection, mixing, or further biological updates, allowing the Lagrangian population to thin naturally in unfavorable environments.



### Next steps:
* add growth
* add medium level and baseline mortality
* add Alexandrium catenella and Dinophysis spp.
* add vertical behavior framework - Implement a shared parameterized vertical-movement function (band / diel_band)
    * Pseudo-nitzschia: shallow band
    * Alexandrium: diel vertical migration (shallow band by day, deeper band by night)
    * Dinophysis: mid-depth band (around the pycnocline)

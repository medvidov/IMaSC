import spacy
from spacy import displacy
import pdfkit


colors = {"SPACECRAFT": "#b35f50", "INSTRUMENT":"#509cb3"}
options = {"ents": ["SPACECRAFT","INSTRUMENT"], "colors": colors}

text = """[7] Aura orbits at a height of 705 km, with an orbital
period of 98.8 min. It is a Sun‐synchronous, polar‐orbiting
satellite, crossing the equator at 1345 local
time on its
ascending node [Levelt et al., 2006; Waters et al., 2006].
The Ozone Monitoring Instrument (OMI) was designed to
measure ozone and other trace gases, with a small footprint
and daily global coverage. OMI’s nadir pointing telescope
has a 13 km by 28 km footprint, and swath width of each scan
is 2600 km [Levelt et al., 2006]. Data examined in this study
use the Total Ozone Mapping Spectrometer (TOMS‐v8)
retrieval algorithm. On average, the OMI‐TOMS (hereafter
referred to as OMI) retrievals of total column ozone agree
better than 1% [McPeters et al., 2008; Balis et al., 2007]
with ground‐based Brewer and Dobson spectrometers.

[8] The Microwave Limb Sounder (MLS) scans in the
plane of orbital motion, retrieving individual profiles every
165 km along its track. The vertical resolution is 3 km, and
approximate horizontal resolution is 300 km in the upper
troposphere/lower stratosphere (UT/LS) region. Precision
ranges from 20% at the 215 hPa level to 3–5% in the
stratosphere [Froidevaux et al., 2008; Boyd et al., 2007]. Of
the methods described in Table 1, we consider two products
that use data from OMI and MLS.
2.1.2. Aura‐Derived Troposphere Products

[9] The first product examined is the Trajectory‐enhanced
Tropospheric Ozone Residual (TTOR). The version (1.6) of
the TTOR product used in this study uses 2‐day isentropic
backward and forward trajectories from GEOS‐4 [Bloom
et al., 2005] to map MLS measurements over the Earth’s
surface, followed by interpolation of these values to a 1.25°
× 1° grid. This differs from the version described by
Schoeberl et al. [2007], which used 6 day isentropic forward
trajectories. Stratospheric column ozone is calculated from
the mapped and interpolated MLS measurements, and then
is subtracted from interpolated OMI total column ozone.
The results of this calculation are daily global estimates of
tropospheric ozone."""

nlp = spacy.load("models")
doc = nlp(text)
nlp.to_disk("documentation")
# displacy.serve(doc, style="ent", options=options)
# html = displacy.render(doc, style="ent", options=options)
# pdfkit.from_string(html,'pres.pdf')


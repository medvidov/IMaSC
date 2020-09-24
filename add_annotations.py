import json

spans = []
item = {}
item["start"] = 0
item["end"] = 0
item["token_start"] = 0
item["token_end"] = 0
item["label"] = ""

instruments = [
    "MLS",
    "Microwave Limb Sounder",
    "SMR",
    "Sub-Millimetre Radiometer",
    "Sub-Millimeter Radiometer",
    "ACE-FTS",
    "Fourier Transform Ultra-Violet Spectrometer",
    "SCIAMACHY",
    "Scanning Imaging Absorption Spectrometer for Atmospheric Chartography",
    "MIPAS",
    "Michelson Interferometer for Passive Atmospheric Sounding",
    "GEOS",
    "Goddard Earth Observing System",
    "IASI",
    "Infrared Atmospheric Sounding Interferometer",
    "HWV",
    "Harvard Water Vapor instrument",
    "CFH",
    "Cryogenic Frostpoint Hygrometer",
    "JLH",
    "JPL Laser Hygrometer",
    "OSIRIS",
    "Optical Spectrograph and Infrared Imager System",
    "CLAES",
    "Cryogenic Limb Array Etalon Spectrometer",
    "CALIOP",
    "Cloud-Aerosol Lidar with Orthogonal Polarization",
    "TOMS",
    "total ozone mapping spectrometer",
    "MODIS",
    "Moderate Resolution Imaging Spectroradiometer",
    "OMI",
    "Ozone Monitoring Instrument"
]
spacecraft = ["Aura", "AURA", "Odin", "ODIN", "Envisat", "ENVISAT", "UARS", "Upper Atmosphere Research Satellite"]
models = []
experiments = ["ACE", "Atmospheric Chemistry Experiment"]
locations = [] # DOn't know if I want this one
algorithms  = []

# Take in data from annotations - Copy.jsonl
# for each JSON
# Check if we have spans
# if yes, spans = spans, if no initialize spans
# text = JSON text
# Substring checking

# spans.append(item)
# spans.append(item)


print(spans)

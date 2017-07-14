# lcos_multispot_pattern

Create a multispot pattern via phase modulation using a LCOS-SLM.

![LCOS multispot phase pattern](pattern.png)

The repo contains:

- `patternlib.py`: the library for generating the multispot pattern.
- `pattern_server.py`: a script to start a "pattern server" listening for
  pattern parameters and returning 2D arrays containing the pattern.
  This server is used by [LCOS_LabVIEW](https://github.com/multispot-software/LCOS_LabVIEW)
  to obain LCOS patterns (LabVIEW and python communicate via a local socket).
- `installer.cfg`: configuration for building an NSIS installer for a stand-alone
  installation of `pattern_server.py` using [pynsist](https://github.com/takluyver/pynsist).
  
The other files are just notebooks GUIs for playing around with generating the pattern
and visualizing it using QT5 or jupyter notebook widgets.

# Dependencies

- python 3.5+
- numpy 1.10+
- pyyaml 3.12

# Cite

If you use this code, please cite the paper:

> 48-spot single-molecule FRET setup with periodic acceptor excitation <br>
> A. Ingargiola, M. Segal, A. Gulinatti, I. Rech, I. Labanca, P. Maccagnani, M. Ghioni, S. Weiss, X. Michalet <br>
> *bioRxiv* 156182; **2017**. doi: https://doi.org/10.1101/156182

----
Copyright (C) 2017 The Regents of the University of California, Antonino Ingargiola and contributors.

*This work was supported by NIH grants R01 GM069709 and R01 GM095904.*




# -*- coding: utf-8 -*-
"""
config.py
---------
Central configuration file for the ED Digital Twin.

This is the ONLY file you need to edit before running the model.
Set FORECAST_DATE to the datetime from which you want to run the
12-hour forward simulation.

Authors:
    Nirmani Amarasinghe  (ORCID: 0009-0001-9719-6366)
    Laura Boyle          (ORCID: 0000-0001-9651-1363)
    Adele H. Marshall    (ORCID: 0000-0001-5306-2756)

    Mathematical Science Research Centre, Queen's University Belfast

Usage:
    1. Change FORECAST_DATE to your desired forecast start point.
    2. Run LOS.py first to generate length-of-stay parameters.
    3. Run Digital_Twin_10_minute.py to execute the simulation.
"""

from datetime import datetime

# --------------------------------------------------------------------
# FORECAST DATE
# --------------------------------------------------------------------
# Set this to the date and time you want the digital twin to forecast
# FROM. The model will simulate the next 12 hours from this point.
#
# Format: datetime(YEAR, MONTH, DAY, HOUR, MINUTE)
# Example: datetime(2007, 9, 20, 12, 0) = 20 Sep 2007 at 12:00 noon
# --------------------------------------------------------------------

FORECAST_DATE = datetime(2007, 9, 20, 12, 0)

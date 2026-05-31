# Data sources

Full provenance for every input that feeds the feature matrices. The README carries
a four-line summary; this is the detailed version.

---

**CGM.** I use the [Stelo by Dexcom](https://www.stelo.com) continuous glucose monitor, an over-the-counter sensor that runs about $45 per month. It records a glucose reading every five minutes and syncs to the Stelo app on my phone. The raw export from Stelo's web portal is the starting point for everything here.

**Walking.** I wear a Google Pixel Watch 2 throughout the day. It tracks activity through Fitbit and syncs that data to Google Health Connect on my phone. Stelo reads activity from Health Connect, so walk events show up alongside glucose readings in a single export. I cross-reference those with watch timestamps to get walk start time and duration.

**Meals.** This is the weakest part of the data. About 24 meals were logged directly in the Stelo app at the time I ate using their AI analyzer feature (take a photo and it identifies dish with an estimate which was likely an experimental feature that was shut down). The remaining ~320 were logged initially in a spreadsheet, then migrated to the web app. It is imperfect and I know it.

**Medications.** Jardiance 10mg, Glipizide 5mg x2, and Crestor 10mg around 10 AM daily. Glipizide 5mg x2 again around 4 PM. The medication log is auto-generated as scheduled rows and I manually edit the record when I shift or miss a dose.

**Fasting windows.** I follow a roughly 12-hour overnight fast, 7 PM to 7 AM. On days when I skip breakfast and push the first meal to lunch, the window gets flagged as an intermittent fast automatically based on meal timing.

**Food nutrition.** I built a custom lookup table for the South Indian dishes I actually eat. Mainstream nutrition APIs don't have granular entries for dosa, rasam, or veg biryani as I prepare them, so I assembled estimates from a combination of published sources and portion-size notes.

---

For a column-by-column description of the resulting feature matrices, see [data-dictionary.md](data-dictionary.md).

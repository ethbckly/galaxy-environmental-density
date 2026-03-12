import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
os.chdir(r"C:\Users\eth\Desktop\Uni\Masters proj")

def crossmatch(file1, file2, max_seperation=1.0*u.arcsec):

    print("Current working directory:", os.getcwd())
    print("Files in directory:", os.listdir())
    
    table1 = Table.read("COSMOS2020_CLASSIC_R1_v2.1_p3.fits")
    table2 = Table.read("table1.dat", format="ascii.cds", readme="ReadMe")

    ra1, dec1 = table1['ALPHA_J2000'], table1['DELTA_J2000']
    ra2, dec2 = table2['RAcdeg'], table2['DEcdeg']

    ra1, dec1, ra2, dec2 = np.array(ra1, dtype=float), np.array(dec1, dtype=float), np.array(ra2, dtype=float), np.array(dec2, dtype=float)

    coords1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    coords2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)

    index12, seperation12, _ = coords1.match_to_catalog_sky(coords2)
    index21, seperation21, _ = coords2.match_to_catalog_sky(coords1)

    seperation_mask = (seperation12 < max_seperation) & (index21[index12] == np.arange(len(coords1)))

    matched_1 = table1[seperation_mask]
    matched_2 = table2[index12[seperation_mask]]

    matched_tables = hstack([matched_1, matched_2], table_names=['table_1', 'table_2'])
    matched_separations = seperation12[seperation_mask].arcsec


    print(f"Number of matched rows: {len(matched_tables)}") 
    matched_tables.write("COSMOS_SMOLCIC", format='fits', overwrite=True)

    return matched_tables, matched_separations


file1 = "COSMOS2020_CLASSIC_R1_v2.1_p3.fits"
file2 = "table1.dat"

matched, separations = crossmatch(file1, file2, max_seperation=1*u.arcsec,
                                  )


plt.figure(figsize=(7, 5))
plt.hist(separations, bins=200,range=(0,10), color='royalblue', alpha=0.75, edgecolor='black')
plt.xlim(0, 3)
plt.xlabel('Separation (arcseconds)')
plt.ylabel('Number of matches')
plt.title('Distribution of Crossmatches for Separations')
plt.grid(alpha=0.4)
plt.show()


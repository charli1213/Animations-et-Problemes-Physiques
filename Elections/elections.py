import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rc('axes', axisbelow=True)

# --- CCRS
import cartopy.crs as ccrs
import cartopy._epsg as cepsg
MTM7crs  = cepsg._EPSGProjection(32187)   # Projection MTM7
UTM19crs = cepsg._EPSGProjection(32619)   # Projection UTM19N
PlateCarree = ccrs.PlateCarree()         # Projection Mercator
Orthographic = ccrs.Orthographic(-60,47) # Projection Orthographique

gdbfile = "Election_2021.gdb"

mini = 1
maxi = 10

# Ça prend tous les fichiers 
shapefile = gpd.read_file(gdbfile,layer=3)

shapefile['No_Distric'] = shapefile['No_Distric'].astype(int)
shapefile = shapefile[(shapefile.No_Distric<=maxi)]
shapefile = shapefile[(shapefile.No_Distric>=mini)]
shapefile = shapefile.sort_values('No_section')



xlsxfilename = 'Recensement Conseillers.xlsx'
xlsfile = pd.ExcelFile(xlsxfilename)
dataframes = []
#for i in [1,2,3,4,5] :
for i in range(mini,maxi+1) :
    dataframes += [pd.read_excel(xlsfile,str(i))]
df = pd.concat(dataframes)
df = df.sort_values('Bureau de vote')

# On extraie de données : 
df = df[df['Bureau de vote']>1000]
df['PourcentageTQ'] = 100*df['TQ']/df['Nombre de votes']
shapefile['PourcentageTQ'] = df['PourcentageTQ'].values
df['PourcentageQFF'] = 100*df['QFF']/df['Nombre de votes']
shapefile['PourcentageQFF'] = df['PourcentageQFF'].values
shapefile['Pourcentage_cumul'] = shapefile['PourcentageTQ'] + shapefile['PourcentageQFF']


color_dict = {'Démocratie Québec'        :'greenyellow',
              'Québec 21'                :'lightskyblue',
              'Québec Forte et Fière'    :'mediumpurple',
              'Transition Québec'        :'orange',
              'Égalité'                  :'white',
              'Équipe Marie-Josée Savard':'aquamarine'}
              

def find_winners(df, color_dict = color_dict) :
    winner_array = []
    id_conversion = dict(zip(range(6),df.keys()[1:6]))
    id_conversion[5] = 'Égalité'
    print(id_conversion)
    for i in range(len(df)) :
        line = df.iloc[i][1:6]
        winner_id = np.where(line == line.max())[0]
        if len(winner_id)>1 :
            winner_id = 5
        winner_array += [int(winner_id)]
    df['Winner'] = winner_array
    df['Winner_short'] = df['Winner'].map(id_conversion)
    df['Winner'] = df['Winner_short'].map({'EMJS':'Équipe Marie-Josée Savard',
                                           'QFF':'Québec Forte et Fière',
                                           'QC21':'Québec 21',
                                           'TQ':'Transition Québec',
                                           'DQ':'Démocratie Québec',
                                           'Égalité':'Égalité'})



# On trouve les gagnants : 
find_winners(df)

# On check quels partis sont réellement sur la map, pour notre cmap.
partis_actifs = list(dict.fromkeys(df.Winner.values))
partis_actifs = [p for p in list(color_dict.keys()) if p in partis_actifs]
couleurs = [color_dict[nom_parti] for nom_parti in partis_actifs]


shapefile['Winner'] = df['Winner'].values



# === FIGURE ===
cmap = cm.get_cmap('YlOrRd',20) # Colormap
fig,axes = plt.subplots(nrows=1,ncols=2, figsize=(18,9.5))

import cartopy.io.img_tiles as cimgt
#request1 = cimgt.GoogleTiles(style='street',desired_tile_form='RGBA')
#request2 = cimgt.GoogleTiles(style='satellite')
#axes[0].add_image(request1, 16, alpha=0.5)
#axes[1].add_image(request2, 16, alpha=0.4)


# --- Pourcentage TQ
im1 = shapefile.plot(column='PourcentageTQ',
                     cmap=cmap,
                     ax=axes[0],
                     legend=True,
                     vmin = 0, vmax = 100,
                     )
shapefile.boundary.plot(color='k', ax=axes[0], linewidth = 0.3)

# --- Gagnants des sections de votes
im2 = shapefile.plot(column='Winner',
                     ax=axes[1],
                     legend = True,
                     legend_kwds = {'loc':'lower right'},
                     cmap=colors.ListedColormap(couleurs),
                     )
shapefile.boundary.plot(color='k', ax=axes[1], linewidth = 0.3)



# --- Fine tunning : 
axes[0].set_title('Pourcentage de vote par pôle (Transition Québec) : Taschereau')
axes[1].set_title('Gagnants par sections de vote, Taschereau')
"""
for point, num_section in zip(shapefile.centroid,shapefile.No_section.values) :
    axes[0].text(point.x, point.y, s=num_section)
"""

axes[0].grid()
axes[1].grid()
axes[0].set_yticklabels('')
axes[1].set_yticklabels('')
axes[0].set_xticklabels('')
axes[1].set_xticklabels('')
plt.tight_layout()
plt.show()


shapegeo = shapefile.geometry
# Ça c'est une GeoSeries qui contient tous les polygones.

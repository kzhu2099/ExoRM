from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# Query all confirmed exoplanets
table = NasaExoplanetArchive.query_criteria(
    table = 'PS',
    select = 'pl_name, pl_bmasse, pl_rade, disc_year, pl_controv_flag'
)

# Convert to pandas DataFrame
data = table.to_pandas()

# Display or save to CSV
data.to_csv('exoplanet_data.csv')
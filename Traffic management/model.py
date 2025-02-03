import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import networkx as nx
import folium

traffic_data = pd.read_csv("Traffic management/Banglore_traffic_Dataset.csv")

print(traffic_data.head())

print(traffic_data.isnull().sum())

traffic_data['Average Speed'].fillna(traffic_data['Average Speed'].mean(), inplace=True)
traffic_data['Congestion Level'].fillna(traffic_data['Congestion Level'].mode()[0], inplace=True)

traffic_data['date'] = pd.to_datetime(traffic_data['date'])

traffic_data['day_of_week'] = traffic_data['date'].dt.dayofweek

traffic_data.drop(columns=['date'], inplace=True)

print(traffic_data.head()) 

X = traffic_data[['day_of_week', 'Traffic Volume', 'Congestion Level', 'Road Capacity Utilization']]
y = traffic_data['Average Speed']

X = pd.get_dummies(X, columns=['Congestion Level'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

G = nx.Graph()

locations = traffic_data['Road/Intersection Name'].unique()
for loc1 in locations:
    for loc2 in locations:
        if loc1 != loc2:
            distance = 10  # Assume distance between locations is 10 km
            speed = traffic_data[traffic_data['Road/Intersection Name'] == loc1]['Average Speed'].mean()
            travel_time = distance / speed
            G.add_edge(loc1, loc2, weight=travel_time)


start = "Sarjapur Road"  # Replace with actual start location
end = "Hebbal Flyover"    # Replace with actual end location
shortest_path = nx.shortest_path(G, source=start, target=end, weight='weight')
print(f"Optimal Route: {shortest_path}")


# Create a map centered on Bangalore
bangalore_map = folium.Map(location=[12.9240089, 77.6531102], zoom_start=12)

# Add markers for locations
for loc in shortest_path:
    folium.Marker(
        location=[12.9240089, 77.6531102],  # Replace with actual coordinates
        popup=loc,
        icon=folium.Icon(color="green" if loc == start else "red" if loc == end else "blue")
    ).add_to(bangalore_map)

# Draw the optimal route on the map
route_coords = [[12.9240089, 77.6531102], [13.041021, 77.5897035]]  # Replace with actual coordinates
folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(bangalore_map)

# Save the map
bangalore_map.save("traffic_map.html")
const map = L.map('map').setView([37.7749, -122.4194], 7); // Zoomed in for better regional view

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);

// Trail coordinates
const trails = [
  {
    name: "Cascades Falls Trail",
    coords: [37.3535, -80.5996],
    color: "blue"
  },
  {
    name: "McAfee Knob Trail",
    coords: [37.3806, -80.0890],
    color: "red"
  },
  {
    name: "Blue Ridge Trail",
    coords: [38.0293, -78.4767],
    color: "green"
  }
];

// Add pins to map
trails.forEach(trail => {
  L.circleMarker(trail.coords, {
    color: trail.color,
    radius: 10
  })
    .addTo(map)
    .bindPopup(trail.name);
});

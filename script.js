document.addEventListener("DOMContentLoaded", () => {
  const map = L.map('map').setView([37.5, -80.3], 9); // Centered between trails

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

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

  trails.forEach(trail => {
    L.circleMarker(trail.coords, {
      color: trail.color,
      radius: 10
    })
      .addTo(map)
      .bindPopup(trail.name);
  });

  // Search filter
  document.getElementById('searchInput').addEventListener('input', function (e) {
    const query = e.target.value.toLowerCase();
    const cards = document.querySelectorAll('.trail-card');

    cards.forEach(card => {
      const text = card.textContent.toLowerCase();
      card.style.display = text.includes(query) ? 'block' : 'none';
    });
  });
});

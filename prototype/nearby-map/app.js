(function() {
    'use strict';

    const mapDiv = document.getElementById('map');
    const searchInput = document.getElementById('searchInput');
    const categorySelect = document.getElementById('categorySelect');
    const locateBtn = document.getElementById('locateBtn');
    const infoCard = document.getElementById('infoCard');
    const status = document.getElementById('status');
    const demoBtn = document.getElementById('demoBtn');

    const map = L.map(mapDiv, {
        center: [35.681236, 139.767125],
        zoom: 14
    });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    let currentMarker = null;
    const facilitiesLayer = L.layerGroup().addTo(map);
    let allFacilities = [];
    let lastCoords = null;

    function clearCurrentMarker() {
        if (currentMarker) {
            map.removeLayer(currentMarker);
            currentMarker = null;
        }
    }

    function setCurrentMarker(lat, lng) {
        clearCurrentMarker();
        currentMarker = L.circleMarker([lat, lng], {
            radius: 8,
            color: '#3388ff',
            fillColor: '#3388ff',
            fillOpacity: 0.8
        }).addTo(map);
    }

    function haversineDistance(lat1, lng1, lat2, lng2) {
        const R = 6371;
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLng = (lng2 - lng1) * Math.PI / 180;
        const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLng / 2) ** 2;
        return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }

    function showInfo(facility) {
        infoCard.replaceChildren();
        const name = document.createElement('div');
        name.textContent = facility.name;
        const type = document.createElement('div');
        type.textContent = facility.type === 'supermarket' ? 'スーパー' : 'ガソリンスタンド';
        const distance = document.createElement('div');
        distance.textContent = `距離: ${facility.dist.toFixed(2)} km`;
        infoCard.appendChild(name);
        infoCard.appendChild(type);
        infoCard.appendChild(distance);
        infoCard.hidden = false;
    }

    function showError(message) {
        status.replaceChildren();
        const text = document.createElement('span');
        text.textContent = `エラー: ${message}`;
        const retry = document.createElement('button');
        retry.type = 'button';
        retry.setAttribute('aria-label', '再試行');
        retry.textContent = '再試行';
        retry.addEventListener('click', () => {
            if (lastCoords) fetchOverpass(lastCoords[0], lastCoords[1]);
        });
        status.appendChild(text);
        status.appendChild(retry);
    }

    function filterFacilities() {
        const search = searchInput.value.trim().toLowerCase();
        const category = categorySelect.value;
        return allFacilities.filter((facility) => {
            const matchesCategory = category === 'all' || facility.type === category;
            const matchesSearch = !search || facility.name.toLowerCase().includes(search);
            return matchesCategory && matchesSearch;
        });
    }

    function render() {
        facilitiesLayer.clearLayers();
        const facilities = filterFacilities();
        if (facilities.length === 0) {
            status.textContent = '該当する施設がありません';
            return;
        }
        facilities.forEach((facility) => {
            const color = facility.type === 'supermarket' ? '#16a34a' : '#f97316';
            const marker = L.circleMarker([facility.lat, facility.lon], {
                radius: 8,
                color,
                fillColor: color,
                fillOpacity: 0.8
            });
            marker.on('click', () => showInfo(facility));
            facilitiesLayer.addLayer(marker);
        });
    }

    function fetchOverpass(lat, lng) {
        lastCoords = [lat, lng];
        status.textContent = '施設情報を読み込み中...';
        const query = `[out:json];(node["shop"="supermarket"](around:3000,${lat},${lng});node["amenity"="fuel"](around:3000,${lat},${lng}););out body;`;
        fetch('https://overpass.kumi.systems/api/interpreter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: `data=${encodeURIComponent(query)}`
        })
            .then((response) => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.json();
            })
            .then((data) => {
                allFacilities = (data.elements || []).filter((element) => element.lat && element.lon).map((element) => {
                    const name = element.tags && element.tags.name ? element.tags.name : '名称未設定';
                    const type = element.tags && element.tags.shop === 'supermarket' ? 'supermarket' : 'fuel';
                    const facilityLat = element.lat;
                    const facilityLon = element.lon;
                    const dist = haversineDistance(lastCoords[0], lastCoords[1], facilityLat, facilityLon);
                    return { name, type, lat: facilityLat, lon: facilityLon, dist };
                });
                render();
                status.textContent = `${allFacilities.length}件の施設が見つかりました`;
            })
            .catch((error) => showError(error.message));
    }

    function handleGeoSuccess(position) {
        const lat = position.coords.latitude;
        const lng = position.coords.longitude;
        setCurrentMarker(lat, lng);
        map.setView([lat, lng], 14);
        demoBtn.style.display = 'none';
        fetchOverpass(lat, lng);
    }

    function handleGeoError() {
        status.textContent = '位置情報を取得できませんでした';
        demoBtn.style.display = 'inline-block';
    }

    function locateUser() {
        if (!navigator.geolocation) {
            handleGeoError();
            return;
        }
        navigator.geolocation.getCurrentPosition(handleGeoSuccess, handleGeoError, {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        });
    }

    locateBtn.addEventListener('click', locateUser);
    demoBtn.addEventListener('click', () => {
        const lat = 35.681236;
        const lng = 139.767125;
        setCurrentMarker(lat, lng);
        map.setView([lat, lng], 14);
        demoBtn.style.display = 'none';
        fetchOverpass(lat, lng);
    });
    searchInput.addEventListener('input', render);
    categorySelect.addEventListener('change', render);
    locateBtn.click();
})();

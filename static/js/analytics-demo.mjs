// Deterministic, illustrative data. Imported only on loopback preview hosts.
function distribute(total, weights) {
  const weight = weights.reduce((sum, value) => sum + value, 0);
  const shares = weights.map(value => total * value / weight);
  const counts = shares.map(Math.floor);
  const remainder = total - counts.reduce((sum, value) => sum + value, 0);
  const order = shares.map((share, index) => ({ index, fraction: share - counts[index] }))
    .sort((a, b) => b.fraction - a.fraction || a.index - b.index);
  for (let index = 0; index < remainder; index++) counts[order[index].index]++;
  return counts;
}

export function sampleAnalytics(dayCount, now = Date.now()) {
  // Recent readership favors East Asia. Earlier visits include a larger
  // European readership and additional locations in the southern hemisphere.
  const places = [
    ['CN', 'Beijing', 40, 116, 34, 2], ['CN', 'Shanghai', 31, 121, 24, 2],
    ['US', 'San Francisco', 38, -122, 18, 6], ['US', 'New York', 41, -74, 5, 17],
    ['JP', 'Tokyo', 36, 140, 6, 3], ['SG', 'Singapore', 1, 104, 5, 3],
    ['KR', 'Seoul', 38, 127, 4, 2], ['GB', 'London', 51, 0, 2, 14],
    ['DE', 'Berlin', 53, 13, 1, 10], ['CA', 'Toronto', 44, -79, 1, 5],
    ['FR', 'Paris', 49, 2, 0, 9], ['AU', 'Sydney', -34, 151, 0, 5],
    ['CH', 'Zurich', 47, 9, 0, 6], ['NL', 'Amsterdam', 52, 5, 0, 5],
    ['IN', 'Bengaluru', 13, 78, 0, 8], ['BR', 'São Paulo', -24, -47, 0, 6],
    ['ZA', 'Johannesburg', -26, 28, 0, 5], ['KE', 'Nairobi', -1, 37, 0, 3],
    ['MX', 'Mexico City', 19, -99, 0, 2],
  ];
  const locationViews = places.map(() => 0);
  const days = Array.from({ length: dayCount }, (_, index) => {
    const offset = dayCount - 1 - index;
    const views = Math.round(86 + (90 - offset) * 1.3 + Math.sin(offset * 1.7) * 39 + (offset % 13 === 2 ? 190 : 0));
    const counts = distribute(views, places.map(place => place[offset < 30 ? 4 : 5]));
    counts.forEach((count, place) => { locationViews[place] += count; });
    return {
      date: new Date(now - offset * 86400000).toISOString().slice(0, 10),
      views,
      visitors: Math.round(views * (0.48 + (offset % 5) * 0.025)),
    };
  });
  const views = days.reduce((sum, day) => sum + day.views, 0);
  const countries = new Map();
  const points = places.map(([country, city, lat, lng], index) => ({ country, city, lat, lng, views: locationViews[index] }))
    .filter(point => point.views > 0);
  for (const point of points) countries.set(point.country, (countries.get(point.country) || 0) + point.views);
  const allocate = (items, key) => {
    const counts = distribute(views, items.map(item => item[1]));
    return items.map(([label], index) => ({ [key]: label, views: counts[index] }));
  };
  return {
    generatedAt: new Date(now).toISOString(),
    startedAt: new Date(now - 180 * 86400000).toISOString(),
    period: { days: dayCount, from: days[0].date, to: days.at(-1).date, views, visitors: Math.round(views * 0.43), countries: countries.size },
    lifetime: { views: 32418, visitors: 12703 },
    today: days.at(-1),
    days,
    countries: Array.from(countries, ([code, count]) => ({ code, views: count })).sort((a, b) => b.views - a.views),
    cities: points.map(({ city, country, views }) => ({ city, country, views })).sort((a, b) => b.views - a.views),
    points,
    pages: allocate([['/', 0.31], ['/research/', 0.19], ['/blog/vla-models-review/', 0.15], ['/papers/', 0.13], ['/blog/oxygen-mot-vla-inference/', 0.10], ['/about/', 0.07], ['/outside/', 0.05]], 'path'),
    referrers: allocate([['', 0.39], ['google.com', 0.27], ['github.com', 0.16], ['scholar.google.com', 0.09], ['zhihu.com', 0.05], ['bing.com', 0.04]], 'host'),
  };
}

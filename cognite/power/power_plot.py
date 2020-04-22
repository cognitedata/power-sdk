import math
from collections import defaultdict

import networkx as nx
import plotly.graph_objects as go
import pyproj
from numpy import mean, nanmean

# unified plotting colors
_MARKER_EDGE_COLOR = "rgb(85,150,210)"
_MARKER_FILL_COLOR = "rgb(230,230,230)"

# univeral transverse mercator zone 32 = south norway, germany
_LATLON_PROJ = "+proj=utm +zone=32, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
_PROJECTION = pyproj.Proj(_LATLON_PROJ, preserve_units=True)


def _latlon_to_xy(lat, lon):
    (x, y) = (lat, lon)
    return (x, y)


def voltage_color(bv):
    color_map = [
        (-1e9, "000000"),
        (100, "000000"),
        (132, "9ACA3C"),
        (300, "20B3DE"),
        (420, "ED1C24"),
        (1e9, "ED1C24"),
    ]
    color_map = [(v, tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))) for v, h in color_map]  # to rgb
    ix_above = 0
    while color_map[ix_above][0] < bv:
        ix_above += 1
    t = (bv - color_map[ix_above - 1][0]) / (color_map[ix_above][0] - color_map[ix_above - 1][0])
    color = [
        int(color_map[ix_above - 1][1][rgb] + t * (color_map[ix_above][1][rgb] - color_map[ix_above - 1][1][rgb]))
        for rgb in range(3)
    ]
    c = ",".join(map(str, color))
    return f"rgb({c})"


def node_locations(power_area, interpolate_missing_positions=True):
    node_loc = {
        name: [
            float(substation.metadata.get("PositionPoint.xPosition", math.nan)),
            float(substation.metadata.get("PositionPoint.yPosition", math.nan)),
        ]
        for name, substation in power_area._graph.nodes(data="object")
    }
    if interpolate_missing_positions:
        orphan_count = 0
        for it in range(2):
            for s, loc in node_loc.items():
                if math.isnan(loc[0]):
                    nb_locs = [
                        node_loc[n] for n in nx.neighbors(power_area._graph, s) if not math.isnan(node_loc[n][0])
                    ]
                    mean_loc = [sum(c) / len(nb_locs) for c in zip(*nb_locs)]
                    if len(mean_loc) == 2:
                        node_loc[s] = mean_loc
                    elif it == 1:
                        node_loc[s] = [20, 55 + orphan_count]  # TODO don't hardcode this
                        orphan_count += 1
    return node_loc


def create_substation_plot(node_locations, node_plot_mode):
    text, x, y = zip(*[(k, v[0], v[1]) for k, v in node_locations.items()])
    return go.Scatter(
        x=x,
        y=y,
        text=text,
        mode=node_plot_mode,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=15, line=dict(color=_MARKER_EDGE_COLOR, width=2), color=_MARKER_FILL_COLOR),
    )


def create_substation_map_plot(node_locations):
    text, lon, lat = zip(*[(k, v[0], v[1]) for k, v in node_locations.items()])
    # to get an edge color we plot the same data twice with difference marker size
    plots = [
        go.Scattermapbox(lat=lat, lon=lon, showlegend=False, marker=dict(size=17, color=_MARKER_EDGE_COLOR),),
        go.Scattermapbox(
            lat=lat,
            lon=lon,
            text=text,
            mode="markers",
            showlegend=False,
            hoverinfo="text",
            marker=dict(size=13, color=_MARKER_FILL_COLOR),
            textposition="top center",
        ),
    ]
    return plots


def edge_locations(power_area, node_locations):
    # there is a gotcha here that having 100s of line plots is resource intensive, so making one for each
    # ac line segment causes computers to catch fire. To get the coloring right we create one for each
    # base voltage value, and then we split the line by adding nans. This makes the function unintuitive.
    networkx_edges = power_area._graph.edges(data=True)
    lons = defaultdict(list)
    lats = defaultdict(list)
    center_lons = defaultdict(list)
    center_lats = defaultdict(list)
    text = defaultdict(list)
    for acls in networkx_edges:
        lon, lat = zip(*[node_locations[s] for s in acls[:2]])
        base_voltage = acls[2]["object"].metadata.get("BaseVoltage_nominalVoltage", "0")
        lats[base_voltage] += list(lat) + [math.nan]
        lons[base_voltage] += list(lon) + [math.nan]
        center_lons[base_voltage].append(mean(lon))
        center_lats[base_voltage].append(mean(lat))
        text[base_voltage].append("{}: {} kV".format(acls[2]["object"].name, base_voltage))

    return lats, lons, center_lats, center_lons, text


def create_line_segment_plot(x, y, center_x, center_y, text):
    line_plots = [
        go.Scatter(
            x=x[base_voltage],
            y=y[base_voltage],
            line=dict(width=2, color=voltage_color(float(base_voltage))),
            hoverinfo="none",
            mode="lines",
        )
        for base_voltage in x.keys()
    ]
    center_plots = [
        go.Scatter(
            x=center_x[base_voltage],
            y=center_y[base_voltage],
            text=text[base_voltage],
            mode="markers",
            hoverinfo="text",
            marker=dict(size=0.0001, color=voltage_color(float(base_voltage))),
        )
        for base_voltage in text.keys()
    ]
    return line_plots + center_plots


def create_line_segment_map_plot(lats, lons, center_lats, center_lons, text):
    line_plots = [
        go.Scattermapbox(
            mode="lines",
            lon=lons[base_voltage],
            lat=lats[base_voltage],
            hoverinfo="none",
            showlegend=False,
            line=dict(color=voltage_color(float(base_voltage)), width=6),
        )
        for base_voltage in lats.keys()
    ]
    center_plots = [
        go.Scattermapbox(
            lat=center_lats[base_voltage],
            lon=center_lons[base_voltage],
            text=text[base_voltage],
            mode="markers",
            showlegend=False,
            hoverinfo="text",
            marker=dict(size=0.0001, color=voltage_color(float(base_voltage))),
        )
        for base_voltage in text.keys()
    ]
    return line_plots + center_plots


class PowerPlot:
    @staticmethod
    def draw_with_map(power_area, height=None):
        # plot substations
        node_locs = node_locations(power_area, interpolate_missing_positions=False)
        substation_plots = create_substation_map_plot(node_locs)

        # plot ac line segments
        lats, lons, center_lats, center_lons, text = edge_locations(power_area, node_locs)
        ac_line_segment_plots = create_line_segment_map_plot(lats, lons, center_lats, center_lons, text)

        center = nanmean([v for v in node_locs.values()], axis=0)
        fig = go.Figure(
            # ordering matters here: substations last so they are drawn on top
            data=ac_line_segment_plots + substation_plots,
            layout=go.Layout(
                hovermode="closest",
                mapbox_style="stamen-terrain",
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=height,
                mapbox=dict(zoom=7, center=dict(lon=center[0], lat=center[1])),
            ),
        )

        return fig

    @staticmethod
    def draw(power_area, labels="fixed", position="project", height=None):
        if position == "source":
            node_positions = node_locations(power_area)
        elif position == "project":
            node_positions = {n: _latlon_to_xy(*xy) for n, xy in node_locations(power_area).items()}
        elif position == "spring":
            node_positions = nx.spring_layout(power_area._graph)
        elif position == "kamada":
            node_positions = nx.kamada_kawai_layout(power_area._graph)
        else:
            raise ValueError(f"Unknown layout {position}")

        node_plot_mode = "markers"
        if labels == "fixed":
            node_plot_mode += "+text"

        # plot substations
        substation_plot = create_substation_plot(node_positions, node_plot_mode)

        # plot ac line segments
        lats, lons, center_lats, center_lons, text = edge_locations(power_area, node_positions)
        ac_line_segment_plots = create_line_segment_plot(lons, lats, center_lons, center_lats, text)

        fig = go.Figure(
            data=ac_line_segment_plots + [substation_plot],
            layout=go.Layout(
                height=height,
                plot_bgcolor="rgb(250,250,250)",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
            ),
        )
        return fig

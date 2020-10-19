import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyproj
from numpy import mean, nanmean

from cognite.power.data_classes import PowerAssetList

# unified plotting colors
_MARKER_EDGE_COLOR = "rgb(85,150,210)"
_MARKER_FILL_COLOR = "rgb(230,230,230)"

# univeral transverse mercator zone 32 = south norway, germany
_LATLON_PROJ = "+proj=utm +zone=32, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
_PROJECTION = pyproj.Proj(_LATLON_PROJ, preserve_units=True)


def _latlon_to_xy(lat, lon):
    (x, y) = (lat, lon)
    return (x, y)


def voltage_color(base_voltage: float):
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
    while color_map[ix_above][0] < base_voltage:
        ix_above += 1
    t = (base_voltage - color_map[ix_above - 1][0]) / (color_map[ix_above][0] - color_map[ix_above - 1][0])
    color = [
        int(color_map[ix_above - 1][1][rgb] + t * (color_map[ix_above][1][rgb] - color_map[ix_above - 1][1][rgb]))
        for rgb in range(3)
    ]
    c = ",".join(map(str, color))
    return f"rgb({c})"


def _flow_color(flow: float):
    return voltage_color(base_voltage=flow)


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


def node_layout(power_area, position):
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
    return node_positions


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

    counter = Counter([(edge[0], edge[1]) for edge in list(power_area._graph.edges(data=True))])
    dups = {key: 1 for key in counter if counter[key] + counter[key[::-1]] == 2}  # TODO: handle 3?
    for acls in networkx_edges:
        lon, lat = zip(*[node_locations[s] for s in acls[:2]])
        center_lat = mean(lat)
        center_lon = mean(lon)
        if (acls[0], acls[1]) in dups:
            # probably there are more elegant ways, but we want to offset the center in cases where there are multiple
            # lines between two substations
            lat_len = abs(lat[1] - lat[0])
            lon_len = abs(lon[1] - lon[0])
            edge_length = math.sqrt((lat_len) ** 2 + (lon_len) ** 2)
            center_lat += 0.005 * dups[(acls[0], acls[1])] * lon_len / (edge_length + 1e-6)
            center_lon += 0.005 * dups[(acls[0], acls[1])] * lat_len / (edge_length + 1e-6)
            dups[(acls[0], acls[1])] *= -1
        base_voltage = acls[2]["object"].metadata.get("BaseVoltage_nominalVoltage", "0")
        lats[base_voltage] += [lat[0], center_lat, lat[1], math.nan]
        lons[base_voltage] += [lon[0], center_lon, lon[1], math.nan]
        center_lons[base_voltage].append(center_lon)
        center_lats[base_voltage].append(center_lat)
        text[base_voltage].append("{}: {} kV".format(acls[2]["object"].name, base_voltage))

    return lats, lons, center_lats, center_lons, text


def create_line_segment_plot(x, y, center_x, center_y, text):
    line_plots = [
        go.Scatter(
            x=x[base_voltage],
            y=y[base_voltage],
            line=dict(width=2, color=voltage_color(float(base_voltage)), shape="spline", smoothing=1.3),
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
    return line_plots, center_plots


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


def _np_datetime_to_ms(np_datetime):
    return np_datetime.astype("datetime64[ms]").astype("uint64")


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
    def draw(power_area, labels="fixed", position="kamada", height=None):
        node_positions = node_layout(power_area, position)

        node_plot_mode = "markers"
        if labels == "fixed":
            node_plot_mode += "+text"

        # plot substations
        substation_plot = create_substation_plot(node_positions, node_plot_mode)

        # plot ac line segments
        lats, lons, center_lats, center_lons, text = edge_locations(power_area, node_positions)
        ac_line_segment_plots, ac_line_label_point_plot = create_line_segment_plot(
            lons, lats, center_lons, center_lats, text
        )

        fig = go.Figure(
            data=ac_line_segment_plots + ac_line_label_point_plot + [substation_plot],
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

    @staticmethod
    def draw_flow(
        power_area,
        labels="fixed",
        position="kamada",
        height=None,
        timeseries_type="estimated_value",
        granularity="1h",
        date: "np.datetime64" = None,
    ):
        """
        Draws power flow through the area.

        Args:
            labels,position,height: as in `draw`
            timeseries_type: type of time series to retrieve, i.e. value/estimated_value.
            granularity: time step at which to average values over, as in the Python SDK `retrieve_dataframe` function.
            date: datetime object at which to visualize flow, use None for now.
        """
        node_plot_mode = "markers"
        if labels == "fixed":
            node_plot_mode += "+text"

        node_positions = node_layout(power_area, position)
        substation_plot = create_substation_plot(node_positions, node_plot_mode)
        lats, lons, center_lats, center_lons, text = edge_locations(power_area, node_positions)
        ac_line_segment_plots, ac_line_label_point_plot = create_line_segment_plot(
            lons, lats, center_lons, center_lats, text
        )

        terminals = PowerAssetList(
            list(set(sum([list(data["terminals"].values()) for f, t, data in power_area._graph.edges(data=True)], []))),
            cognite_client=power_area._cognite_client,
        )
        ts = terminals.time_series(measurement_type="ThreePhaseActivePower", timeseries_type=timeseries_type)
        analogs = power_area._cognite_client.assets.retrieve_multiple(ids=[t.asset_id for t in ts])
        terminal_ids: List[int] = [a.parent_id for a in analogs]

        target_time = np.datetime64(date or datetime.now())
        delta = np.timedelta64(5, "D")
        start = _np_datetime_to_ms((target_time - delta))
        end = _np_datetime_to_ms((target_time + delta))
        df = power_area._cognite_client.datapoints.retrieve_dataframe(
            id=[t.id for t in ts],
            aggregates=["average"],
            granularity=granularity,
            start=start,  # TODO: split data prep and update
            end=end,
            include_aggregate_name=False,
        )
        df.columns = terminal_ids

        ix = np.searchsorted(df.index, target_time, side="left")
        flow_values = df.iloc[ix - 1, :]
        title = f"flow at {df.index[ix - 1]}"

        distances = [
            np.linalg.norm(np.array(node_positions[edge[0]]) - np.array(node_positions[edge[1]]))
            for edge in power_area._graph.edges
        ]
        global_arrow_scale = 0.15 * np.mean(distances)  # TODO: what is reasonable here?

        arrow_traces = []
        for f, t, data in power_area._graph.edges(data=True):
            terminal_map = data["terminals"]
            terminals = [terminal_map[f], terminal_map[t]]

            flow_values_t = []
            for side in [0, 1]:
                val = np.nan
                if terminals[side].id in flow_values.index:
                    val = flow_values[terminals[side].id]
                    if isinstance(val, pd.Series):
                        val = val.dropna()
                        val = val.mean() if not val.empty else np.nan
                flow_values_t.append(val)

            from_pos = np.array(node_positions[f])
            to_pos = np.array(node_positions[t])
            from_to_vec = to_pos - from_pos

            distance = np.linalg.norm(from_to_vec)
            arrow_scale = min(global_arrow_scale, 0.3 * distance)

            from_to_vec /= max(distance, 0.1)

            if flow_values_t[0] < flow_values_t[1]:
                flow_vec = -from_to_vec
            else:
                flow_vec = from_to_vec
            orthogonal = np.array([-flow_vec[1], flow_vec[0]])

            mid = (from_pos + to_pos) / 2

            sign_from = math.copysign(1, flow_values_t[0]) if not np.isnan(flow_values_t[0]) else 0
            arrow_from_mid = mid - 0.5 * arrow_scale * from_to_vec  # arrow middle is always closer to from
            # direction of arrow depends on sign of flow
            arrow_from_tail = arrow_from_mid - 0.33 * arrow_scale * flow_vec * sign_from
            arrow_from_head = arrow_from_mid + 0.33 * arrow_scale * flow_vec * sign_from
            arrow_from_left = arrow_from_tail - orthogonal * global_arrow_scale * 0.5
            arrow_from_right = arrow_from_tail + orthogonal * global_arrow_scale * 0.5

            sign_to = math.copysign(1, flow_values_t[1]) if not np.isnan(flow_values_t[1]) else 0
            arrow_to_mid = mid + 0.5 * arrow_scale * from_to_vec  # arrow middle is always closer to to
            # direction of arrow depends on sign of flow
            arrow_to_tail = arrow_to_mid - 0.33 * arrow_scale * flow_vec * (-sign_to)
            arrow_to_head = arrow_to_mid + 0.33 * arrow_scale * flow_vec * (-sign_to)
            arrow_to_left = arrow_to_tail - orthogonal * global_arrow_scale * 0.5
            arrow_to_right = arrow_to_tail + orthogonal * global_arrow_scale * 0.5

            arrows = [
                [arrow_from_left, arrow_from_head, arrow_from_right],
                [arrow_to_left, arrow_to_head, arrow_to_right],
            ]
            for arrow_points, terminal, flow in zip(arrows, terminals, flow_values_t):
                arrow_x, arrow_y = zip(*arrow_points)
                arrow_traces.append(  # this makes computers go on fire, but not sure how to fix that.
                    go.Scatter(
                        x=arrow_x,
                        y=arrow_y,
                        text=f"{terminal.name}: {flow:.1f} MW" if not np.isnan(flow) else f"{terminal.name}: NO DATA",
                        line=dict(
                            width=2,
                            color=_flow_color(abs(flow)) if flow and not np.isnan(flow) else "rgb(200,200,200)",
                        ),
                        hoverinfo="text",
                        mode="lines",
                    )
                )
        fig = go.Figure(
            data=ac_line_segment_plots + [substation_plot] + arrow_traces,
            layout=go.Layout(
                title=title,
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

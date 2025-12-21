"""
Traffic Visualization Module
Advanced visualization tools for traffic analysis results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2


class TrafficVisualizer:
    """Comprehensive traffic visualization toolkit"""

    def __init__(self, style: str = "default"):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style to use (default, ggplot, bmh, etc.)
        """
        try:
            # Try seaborn-v0_8 style names for newer matplotlib versions
            if style == "seaborn":
                style = "seaborn-v0_8"
            plt.style.use(style)
        except OSError:
            # Fall back to default if style not found
            plt.style.use("default")

        sns.set_palette("husl")

    def plot_vehicle_counts_over_time(
        self, traffic_data: List[Dict], save_path: str = None
    ) -> plt.Figure:
        """Plot vehicle counts over time"""
        df = pd.DataFrame(traffic_data)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot total vehicles
        ax.plot(
            df.index,
            df["total_vehicles"],
            linewidth=2,
            label="Total Vehicles",
            color="blue",
        )

        # Plot by vehicle type if available
        if "vehicle_counts" in df.columns and not df["vehicle_counts"].empty:
            vehicle_types = set()
            for counts in df["vehicle_counts"]:
                if isinstance(counts, dict):
                    vehicle_types.update(counts.keys())

            for vehicle_type in vehicle_types:
                counts = [
                    counts.get(vehicle_type, 0) if isinstance(counts, dict) else 0
                    for counts in df["vehicle_counts"]
                ]
                ax.plot(df.index, counts, label=vehicle_type.capitalize(), alpha=0.7)

        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("Vehicle Counts Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_congestion_levels(
        self, traffic_data: List[Dict], save_path: str = None
    ) -> plt.Figure:
        """Plot congestion levels over time"""
        df = pd.DataFrame(traffic_data)

        # Map congestion levels to numbers
        congestion_map = {"low": 1, "medium": 2, "high": 3}
        congestion_numeric = [
            congestion_map.get(level, 0) for level in df["congestion_level"]
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Time series plot
        colors = ["green", "yellow", "red"]
        for i, (level, color) in enumerate(zip(["low", "medium", "high"], colors), 1):
            mask = np.array(congestion_numeric) == i
            if mask.any():
                ax1.scatter(
                    df.index[mask],
                    np.array(congestion_numeric)[mask],
                    c=color,
                    alpha=0.6,
                    label=level.capitalize(),
                    s=20,
                )

        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Congestion Level")
        ax1.set_title("Congestion Levels Over Time")
        ax1.set_yticks([1, 2, 3])
        ax1.set_yticklabels(["Low", "Medium", "High"])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Distribution pie chart
        congestion_counts = pd.Series(df["congestion_level"]).value_counts()
        colors_pie = ["green", "yellow", "red"]
        ax2.pie(
            congestion_counts.values,
            labels=congestion_counts.index,
            autopct="%1.1f%%",
            colors=colors_pie[: len(congestion_counts)],
        )
        ax2.set_title("Congestion Level Distribution")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_traffic_density_heatmap(
        self, traffic_data: List[Dict], save_path: str = None
    ) -> plt.Figure:
        """Create traffic density heatmap"""
        df = pd.DataFrame(traffic_data)

        # Create time bins (every 100 frames)
        time_bins = np.arange(0, len(df), 100)
        density_matrix = []

        for i in range(len(time_bins) - 1):
            start_idx = time_bins[i]
            end_idx = time_bins[i + 1]
            chunk = df.iloc[start_idx:end_idx]

            density_row = [
                chunk["traffic_density"].mean(),
                chunk["total_vehicles"].mean(),
                len(chunk[chunk["congestion_level"] == "high"]),
                (
                    chunk["average_speed"].mean()
                    if "average_speed" in chunk.columns
                    else 0
                ),
            ]
            density_matrix.append(density_row)

        density_df = pd.DataFrame(
            density_matrix,
            columns=["Density", "Vehicle Count", "High Congestion", "Avg Speed"],
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(density_df.T, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title("Traffic Metrics Heatmap Over Time")
        ax.set_xlabel("Time Bins")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_interactive_dashboard(self, traffic_data: List[Dict]) -> go.Figure:
        """Create interactive Plotly dashboard"""
        df = pd.DataFrame(traffic_data)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Vehicle Counts",
                "Congestion Levels",
                "Traffic Density",
                "Speed Analysis",
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"secondary_y": True}, {"secondary_y": True}],
            ],
        )

        # Vehicle counts
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["total_vehicles"], mode="lines", name="Total Vehicles"
            ),
            row=1,
            col=1,
        )

        # Congestion levels
        congestion_colors = {"low": "green", "medium": "yellow", "high": "red"}
        for level in ["low", "medium", "high"]:
            mask = df["congestion_level"] == level
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=df.index[mask],
                        y=df["total_vehicles"][mask],
                        mode="markers",
                        name=f"{level.capitalize()} Congestion",
                        marker=dict(color=congestion_colors[level]),
                    ),
                    row=1,
                    col=2,
                )

        # Traffic density
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["traffic_density"],
                mode="lines",
                name="Traffic Density",
                line=dict(color="purple"),
            ),
            row=2,
            col=1,
        )

        # Speed analysis (if available)
        if "average_speed" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["average_speed"],
                    mode="lines",
                    name="Average Speed",
                    line=dict(color="orange"),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            height=800, showlegend=True, title_text="Traffic Flow Analysis Dashboard"
        )

        return fig

    def plot_vehicle_type_distribution(
        self, traffic_statistics: Dict, save_path: str = None
    ) -> plt.Figure:
        """Plot vehicle type distribution"""
        vehicle_breakdown = traffic_statistics.get("vehicle_breakdown", {})

        if not vehicle_breakdown:
            print("No vehicle breakdown data available")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        ax1.pie(
            vehicle_breakdown.values(),
            labels=vehicle_breakdown.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax1.set_title("Vehicle Type Distribution")

        # Bar chart
        ax2.bar(vehicle_breakdown.keys(), vehicle_breakdown.values())
        ax2.set_title("Vehicle Count by Type")
        ax2.set_xlabel("Vehicle Type")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_trajectory_visualization(
        self,
        tracked_objects: Dict[int, Dict],
        frame_shape: Tuple[int, int],
        save_path: str = None,
    ) -> plt.Figure:
        """Visualize vehicle trajectories"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set axis limits based on frame shape
        ax.set_xlim(0, frame_shape[1])
        ax.set_ylim(frame_shape[0], 0)  # Invert y-axis for image coordinates

        colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_objects)))

        for i, (obj_id, obj_info) in enumerate(tracked_objects.items()):
            trajectory = obj_info.get("trajectory", [])
            if len(trajectory) > 1:
                x_coords = [pt[0] for pt in trajectory]
                y_coords = [pt[1] for pt in trajectory]

                # Plot trajectory
                ax.plot(
                    x_coords,
                    y_coords,
                    color=colors[i % len(colors)],
                    alpha=0.7,
                    linewidth=2,
                    label=f"Vehicle {obj_id}",
                )

                # Mark start and end points
                ax.scatter(
                    x_coords[0],
                    y_coords[0],
                    color=colors[i % len(colors)],
                    s=100,
                    marker="o",
                    edgecolor="black",
                    linewidth=2,
                )
                ax.scatter(
                    x_coords[-1],
                    y_coords[-1],
                    color=colors[i % len(colors)],
                    s=100,
                    marker="s",
                    edgecolor="black",
                    linewidth=2,
                )

        ax.set_title("Vehicle Trajectories")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_speed_distribution(
        self, speeds: List[float], save_path: str = None
    ) -> plt.Figure:
        """Create speed distribution visualization"""
        if not speeds:
            print("No speed data available")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(speeds, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.axvline(
            np.mean(speeds),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(speeds):.1f}",
        )
        ax1.axvline(
            np.median(speeds),
            color="green",
            linestyle="--",
            label=f"Median: {np.median(speeds):.1f}",
        )
        ax1.set_xlabel("Speed (pixels/frame)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Speed Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(speeds, vert=True)
        ax2.set_ylabel("Speed (pixels/frame)")
        ax2.set_title("Speed Box Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def generate_summary_report(
        self, traffic_statistics: Dict, save_path: str = None
    ) -> str:
        """Generate summary report"""
        report = f"""
# Traffic Flow Analysis Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Analysis Duration: {traffic_statistics.get('analysis_duration', 0):.1f} seconds
- Total Frames Processed: {traffic_statistics.get('total_frames', 0)}
- Total Vehicles Detected: {traffic_statistics.get('total_vehicles_detected', 0)}
- Average Vehicles per Frame: {traffic_statistics.get('average_vehicles_per_frame', 0):.1f}
- Peak Vehicle Count: {traffic_statistics.get('peak_vehicle_count', 0)}

## Vehicle Type Breakdown
"""

        vehicle_breakdown = traffic_statistics.get("vehicle_breakdown", {})
        for vehicle_type, count in vehicle_breakdown.items():
            percentage = (
                (count / sum(vehicle_breakdown.values()) * 100)
                if vehicle_breakdown
                else 0
            )
            report += f"- {vehicle_type.capitalize()}: {count} ({percentage:.1f}%)\n"

        report += f"""
## Traffic Flow Analysis
- Overall Trend: {traffic_statistics.get('traffic_flow_trend', 'Unknown')}
- Average Speed: {traffic_statistics.get('average_speed', 0):.1f} pixels/frame

## Congestion Analysis
"""

        congestion_dist = traffic_statistics.get("congestion_distribution", {})
        total_congestion = sum(congestion_dist.values())
        for level, count in congestion_dist.items():
            percentage = (count / total_congestion * 100) if total_congestion > 0 else 0
            report += f"- {level.capitalize()} Congestion: {count} frames ({percentage:.1f}%)\n"

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)

        return report

"""
Graph generation module for patient data visualization.
Extracts time-series data from patient chunks and generates interactive graphs.
"""
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Prompt for extracting structured time-series data from text chunks
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical data extraction assistant. Extract time-series data points from the provided patient data chunks.

For each data point, extract:
- date: The date when the measurement was taken (format: YYYY-MM-DD or as close as possible)
- value: The numerical value of the measurement
- unit: The unit of measurement (if mentioned)

Return a JSON array of objects with fields: date, value, unit.

If dates are not explicitly mentioned, try to infer relative dates (e.g., "latest", "recent", "initial") or use the metadata date if available.

Example output:
[
  {{"date": "2024-01-15", "value": 65.5, "unit": "kg"}},
  {{"date": "2024-02-20", "value": 66.2, "unit": "kg"}},
  {{"date": "2024-03-10", "value": 65.8, "unit": "kg"}}
]

If you cannot extract any data points, return an empty array [].

Focus on extracting data for: {characteristic}"""),
    ("user", """Patient data chunks:
{chunks}

Extract all time-series data points for {characteristic}.""")
])

# Prompt for extracting categorical data (e.g., test names and values)
CATEGORICAL_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical data extraction assistant. Extract categorical data points from the provided patient data chunks.

For each data point, extract:
- category: The name/label of the test, measurement, or category (e.g., "FSH", "LH", "Estradiol", "Testosterone")
- value: The numerical value of the measurement
- unit: The unit of measurement (if mentioned)

Return a JSON array of objects with fields: category, value, unit.

Example output:
[
  {{"category": "FSH", "value": 5.2, "unit": "mIU/mL"}},
  {{"category": "LH", "value": 3.8, "unit": "mIU/mL"}},
  {{"category": "Estradiol", "value": 125.5, "unit": "pg/mL"}},
  {{"category": "Testosterone", "value": 0.45, "unit": "ng/mL"}}
]

If you cannot extract any data points, return an empty array [].

Focus on extracting data for: {characteristic}"""),
    ("user", """Patient data chunks:
{chunks}

Extract all categorical data points for {characteristic}. Extract test names, lab values, hormone levels, or any named measurements with their corresponding values.""")
])


def extract_time_series_data(chunks: List[str], characteristic: str) -> List[Dict[str, Any]]:
    """
    Extract time-series data from patient chunks using LLM.
    
    Args:
        chunks: List of text chunks containing patient data
        characteristic: The characteristic to extract (e.g., "weight", "height", "blood pressure")
    
    Returns:
        List of dictionaries with date, value, and unit fields
    """
    if not chunks:
        return []
    
    # Combine chunks into a single text
    chunks_text = "\n---\n".join(chunks)
    
    # Use LLM to extract structured data
    chain = EXTRACTION_PROMPT | llm | JsonOutputParser()
    
    try:
        data_points = chain.invoke({
            "chunks": chunks_text,
            "characteristic": characteristic
        })
        
        # Validate and clean data points
        cleaned_points = []
        for point in data_points:
            if isinstance(point, dict) and "value" in point:
                # Try to parse date
                date_str = point.get("date", "")
                if date_str:
                    try:
                        # Try various date formats
                        date = pd.to_datetime(date_str, errors='coerce')
                        if pd.isna(date):
                            # If parsing fails, skip this point
                            continue
                    except:
                        continue
                else:
                    continue
                
                # Extract numeric value
                try:
                    value = float(point["value"])
                except (ValueError, TypeError):
                    continue
                
                cleaned_points.append({
                    "date": date,
                    "value": value,
                    "unit": point.get("unit", "")
                })
        
        return cleaned_points
    except Exception as e:
        print(f"Error extracting time-series data: {e}")
        return []


def extract_categorical_data(chunks: List[str], characteristic: str) -> List[Dict[str, Any]]:
    """
    Extract categorical data (e.g., test names and values) from patient chunks using LLM.
    
    Args:
        chunks: List of text chunks containing patient data
        characteristic: The characteristic to extract (e.g., "hormone profile", "lab results")
    
    Returns:
        List of dictionaries with category, value, and unit fields
    """
    if not chunks:
        return []
    
    # Combine chunks into a single text
    chunks_text = "\n---\n".join(chunks)
    
    # Use LLM to extract structured data
    chain = CATEGORICAL_EXTRACTION_PROMPT | llm | JsonOutputParser()
    
    try:
        data_points = chain.invoke({
            "chunks": chunks_text,
            "characteristic": characteristic
        })
        
        # Validate and clean data points
        cleaned_points = []
        for point in data_points:
            if isinstance(point, dict) and "value" in point and "category" in point:
                # Extract category name
                category = str(point.get("category", "")).strip()
                if not category:
                    continue
                
                # Extract numeric value
                try:
                    value = float(point["value"])
                except (ValueError, TypeError):
                    continue
                
                cleaned_points.append({
                    "category": category,
                    "value": value,
                    "unit": point.get("unit", "")
                })
        
        return cleaned_points
    except Exception as e:
        print(f"Error extracting categorical data: {e}")
        return []


def parse_dates_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract dates from chunk metadata if available.
    """
    dates = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        # Common date fields in metadata
        for date_field in ["date", "Date", "Date_of_birth", "measurement_date", "record_date"]:
            if date_field in metadata:
                try:
                    date_val = pd.to_datetime(metadata[date_field], errors='coerce')
                    if not pd.isna(date_val):
                        dates.append(date_val)
                except:
                    pass
    return dates


def generate_trajectory_graph(
    data_points: List[Dict[str, Any]],
    characteristic: str,
    patient_name: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive trajectory graph using Plotly.
    
    Args:
        data_points: List of dicts with 'date', 'value', and optionally 'unit'
        characteristic: Name of the characteristic being graphed
        patient_name: Optional patient name for title
    
    Returns:
        Plotly figure object
    """
    if not data_points:
        # Return empty graph with message
        fig = go.Figure()
        fig.add_annotation(
            text="No time-series data found for this characteristic.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    df = df.sort_values("date")
    
    # Get unit (use first non-empty unit if available)
    unit = df["unit"].iloc[0] if len(df) > 0 and df["unit"].iloc[0] else ""
    y_label = f"{characteristic.title()}" + (f" ({unit})" if unit else "")
    
    # Create the graph
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["value"],
        mode='lines+markers',
        name=characteristic,
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, color='#1f77b4'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y}<extra></extra>'
    ))
    
    # Update layout
    title = f"{characteristic.title()} Trajectory"
    if patient_name:
        title = f"{patient_name}'s {title}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig


def generate_multi_patient_graph(
    patient_data: Dict[str, List[Dict[str, Any]]],
    characteristic: str
) -> go.Figure:
    """
    Generate a graph comparing multiple patients' trajectories.
    
    Args:
        patient_data: Dictionary mapping patient names to their data points
        characteristic: Name of the characteristic being graphed
    
    Returns:
        Plotly figure object
    """
    if not patient_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data found for comparison.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # Color palette for multiple patients
    colors = px.colors.qualitative.Set3
    
    for idx, (patient_name, data_points) in enumerate(patient_data.items()):
        if not data_points:
            continue
        
        df = pd.DataFrame(data_points)
        df = df.sort_values("date")
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["value"],
            mode='lines+markers',
            name=patient_name,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
            hovertemplate=f'<b>{patient_name}</b><br>Date: %{{x}}<br>Value: %{{y}}<extra></extra>'
        ))
    
    # Get unit from first patient's data
    unit = ""
    for data_points in patient_data.values():
        if data_points and data_points[0].get("unit"):
            unit = data_points[0]["unit"]
            break
    
    y_label = f"{characteristic.title()}" + (f" ({unit})" if unit else "")
    
    fig.update_layout(
        title=f"{characteristic.title()} Comparison",
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def generate_bar_chart(
    data_points: List[Dict[str, Any]],
    characteristic: str,
    patient_name: Optional[str] = None,
    x_axis_label: Optional[str] = None,
    y_axis_label: Optional[str] = None
) -> go.Figure:
    """
    Generate an interactive bar chart using Plotly.
    
    Args:
        data_points: List of dicts with 'category', 'value', and optionally 'unit'
        characteristic: Name of the characteristic being graphed
        patient_name: Optional patient name for title
        x_axis_label: Custom label for x-axis (defaults to "Test Name" or "Category")
        y_axis_label: Custom label for y-axis (defaults to characteristic name)
    
    Returns:
        Plotly figure object
    """
    if not data_points:
        # Return empty graph with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data found for this characteristic.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Sort by value (descending) for better visualization
    df = df.sort_values("value", ascending=False)
    
    # Get unit (use first non-empty unit if available)
    unit = df["unit"].iloc[0] if len(df) > 0 and df["unit"].iloc[0] else ""
    
    # Set axis labels
    x_label = x_axis_label or "Test Name"
    y_label = y_axis_label or characteristic.title()
    if unit:
        y_label += f" ({unit})"
    
    # Create the bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["category"],
        y=df["value"],
        name=characteristic,
        marker=dict(color='#1f77b4'),
        hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
    ))
    
    # Update layout
    title = f"{characteristic.title()}"
    if patient_name:
        title = f"{patient_name}'s {title}"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        height=500,
        showlegend=False,
        xaxis=dict(tickangle=-45)  # Rotate x-axis labels for readability
    )
    
    return fig


def generate_multi_patient_bar_chart(
    patient_data: Dict[str, List[Dict[str, Any]]],
    characteristic: str,
    x_axis_label: Optional[str] = None,
    y_axis_label: Optional[str] = None
) -> go.Figure:
    """
    Generate a grouped bar chart comparing multiple patients.
    
    Args:
        patient_data: Dictionary mapping patient names to their data points
        characteristic: Name of the characteristic being graphed
        x_axis_label: Custom label for x-axis
        y_axis_label: Custom label for y-axis
    
    Returns:
        Plotly figure object
    """
    if not patient_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data found for comparison.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Collect all unique categories across all patients
    all_categories = set()
    for data_points in patient_data.values():
        for point in data_points:
            if "category" in point:
                all_categories.add(point["category"])
    
    all_categories = sorted(list(all_categories))
    
    if not all_categories:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data found.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create data structure for grouped bars
    fig = go.Figure()
    
    # Color palette for multiple patients
    colors = px.colors.qualitative.Set3
    
    for idx, (patient_name, data_points) in enumerate(patient_data.items()):
        if not data_points:
            continue
        
        # Create a dict for quick lookup
        value_dict = {point["category"]: point["value"] for point in data_points if "category" in point and "value" in point}
        
        # Get values for all categories (use 0 if missing)
        values = [value_dict.get(cat, 0) for cat in all_categories]
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Bar(
            name=patient_name,
            x=all_categories,
            y=values,
            marker=dict(color=color),
            hovertemplate=f'<b>{patient_name}</b><br>%{{x}}: %{{y}}<extra></extra>'
        ))
    
    # Get unit from first patient's data
    unit = ""
    for data_points in patient_data.values():
        if data_points and data_points[0].get("unit"):
            unit = data_points[0]["unit"]
            break
    
    x_label = x_axis_label or "Test Name"
    y_label = y_axis_label or characteristic.title()
    if unit:
        y_label += f" ({unit})"
    
    fig.update_layout(
        title=f"{characteristic.title()} Comparison",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        height=500,
        barmode='group',  # Group bars side by side
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        xaxis=dict(tickangle=-45)
    )
    
    return fig


"""
Interactive Calendar Component with Holiday Recognition
------------------------------------------------------
A customized date picker component that shows holidays and special events.
"""
import streamlit as st
import pandas as pd
import datetime
import calendar
from holiday_data import get_holiday_info, COUNTRY_HOLIDAYS, GLOBAL_HOLIDAYS, SPECIAL_EVENTS

def get_month_calendar(year, month, country='global'):
    """
    Generate a calendar dataframe for a specific month with holiday information.
    
    Args:
        year (int): Year
        month (int): Month (1-12)
        country (str): Country for holiday data
        
    Returns:
        pd.DataFrame: Calendar dataframe with holiday information
    """
    # Get number of days in the month
    num_days = calendar.monthrange(year, month)[1]
    
    # Create dataframe for the month
    days = list(range(1, num_days + 1))
    dates = [datetime.date(year, month, day) for day in days]
    weekdays = [date.strftime('%a') for date in dates]
    
    # Get holiday information for each day
    holiday_names = []
    holiday_factors = []
    
    for date in dates:
        holiday_info = get_holiday_info(date, country)
        if holiday_info['name'] == 'No Holiday':
            holiday_names.append(None)
            holiday_factors.append(1.0)
        else:
            holiday_names.append(holiday_info['name'])
            holiday_factors.append(holiday_info['demand_factor'])
    
    # Create dataframe
    calendar_df = pd.DataFrame({
        'day': days,
        'date': dates,
        'weekday': weekdays,
        'holiday': holiday_names,
        'demand_factor': holiday_factors
    })
    
    # Add week number for display
    calendar_df['week'] = [date.isocalendar()[1] for date in dates]
    
    return calendar_df

def color_holidays(val, demand_factor):
    """Function to color cells based on holiday status"""
    color = 'white'
    text_color = 'black'
    
    if pd.notnull(val):  # This is a holiday
        if demand_factor >= 1.7:  # Major holiday
            color = '#ffcccc'  # Light red
            text_color = '#990000'  # Dark red
        elif demand_factor >= 1.4:  # Medium holiday
            color = '#ffffcc'  # Light yellow
            text_color = '#b35900'  # Brown
        else:  # Minor holiday
            color = '#e6f3ff'  # Light blue
            text_color = '#0066cc'  # Blue
            
    return f'background-color: {color}; color: {text_color}'

def interactive_calendar(default_date=None, country='global', key_prefix='cal'):
    """
    Display an interactive calendar with holiday information.
    
    Args:
        default_date (datetime.date, optional): Default selected date
        country (str): Country for holiday information
        key_prefix (str): Prefix for Streamlit widget keys
        
    Returns:
        datetime.date: Selected date
    """
    if default_date is None:
        default_date = datetime.date.today()
    
    # Get current month and year from default date
    current_month = default_date.month
    current_year = default_date.year
    
    # Create month navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("◀ Prev", key=f"{key_prefix}_prev"):
            if current_month == 1:
                current_month = 12
                current_year -= 1
            else:
                current_month -= 1
    
    with col2:
        st.markdown(f"### {calendar.month_name[current_month]} {current_year}")
    
    with col3:
        if st.button("Next ▶", key=f"{key_prefix}_next"):
            if current_month == 12:
                current_month = 1
                current_year += 1
            else:
                current_month += 1
    
    # Generate calendar dataframe
    calendar_df = get_month_calendar(current_year, current_month, country)
    
    # Reshape for display (each row is a week)
    weeks = calendar_df['week'].unique()
    
    # Create a calendar view
    st.write("**Select a date from the calendar:**")
    
    # Add some CSS styling
    st.markdown("""
    <style>
    .calendar-day {
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .calendar-day:hover {
        background-color: #f0f0f0;
    }
    .holiday {
        font-weight: bold;
    }
    .weekend {
        background-color: #f9f9f9;
    }
    .current-day {
        border: 2px solid #4CAF50;
    }
    .selected-day {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create weekday headers
    cols = st.columns(7)
    for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
        cols[i].markdown(f"<div class='calendar-day'><b>{day}</b></div>", unsafe_allow_html=True)
    
    # Track which date was selected
    selected_date = None
    
    # Display calendar by week
    for week in weeks:
        week_data = calendar_df[calendar_df['week'] == week]
        
        # If this is the first week, we might need to pad with empty cells
        first_day_weekday = week_data.iloc[0]['date'].weekday()  # 0 = Monday, 6 = Sunday
        
        cols = st.columns(7)
        
        # Add empty cells for the first week if needed
        for i in range(first_day_weekday):
            cols[i].write("")
        
        # Fill in the days
        for _, row in week_data.iterrows():
            day = row['day']
            date = row['date']
            weekday = row['weekday']
            holiday = row['holiday']
            demand_factor = row['demand_factor']
            
            # Determine cell classes
            classes = ['calendar-day']
            if weekday in ['Sat', 'Sun']:
                classes.append('weekend')
            if holiday:
                classes.append('holiday')
            if date == datetime.date.today():
                classes.append('current-day')
            
            # Determine column index (Monday=0, Sunday=6)
            col_idx = date.weekday()
            
            # Create calendar cell
            with cols[col_idx]:
                # Create a button for each day
                if st.button(
                    f"{day}",
                    key=f"{key_prefix}_{date.strftime('%Y%m%d')}",
                    help=holiday if holiday else f"{date.strftime('%A, %B %d, %Y')}"
                ):
                    selected_date = date
                
                # Show holiday indicator if applicable
                if holiday:
                    impact = int((demand_factor - 1) * 100)
                    st.markdown(
                        f"<div style='font-size:0.7em; color: {'red' if impact > 50 else 'orange' if impact > 30 else 'blue'}; text-align: center;'>+{impact}%</div>",
                        unsafe_allow_html=True
                    )
    
    # If a date was selected, return it, otherwise return the default date
    return selected_date if selected_date else default_date

def holiday_calendar_picker(title="Select Date", default_date=None, countries=None, key_prefix="cal"):
    """
    Display a holiday calendar picker component.
    
    Args:
        title (str): Title for the component
        default_date (datetime.date, optional): Default date
        countries (list, optional): List of available countries
        key_prefix (str): Prefix for Streamlit widget keys
        
    Returns:
        tuple: (selected_date, selected_country)
    """
    st.write(f"## {title}")
    
    if countries is None:
        # Get countries from holiday data
        countries = ['global'] + sorted(list(COUNTRY_HOLIDAYS.keys()))
    
    # Country selection
    selected_country = st.selectbox(
        "Select Country/Region for Holiday Information",
        countries,
        index=0,
        key=f"{key_prefix}_country"
    )
    
    # Display calendar
    selected_date = interactive_calendar(default_date, selected_country, key_prefix)
    
    # Display selected date information
    holiday_info = get_holiday_info(selected_date, selected_country)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Selected Date:** {selected_date.strftime('%A, %B %d, %Y')}")
        
        if holiday_info['name'] != 'No Holiday':
            st.success(f"**Holiday:** {holiday_info['name']}")
            st.write(f"**Demand Impact:** +{int((holiday_info['demand_factor'] - 1) * 100)}%")
        else:
            st.write("No holidays or special events on this date.")
    
    # Look for special events too
    with col2:
        year = selected_date.year
        month = selected_date.month
        day = selected_date.day
        
        if year in SPECIAL_EVENTS:
            for event in SPECIAL_EVENTS[year]:
                if event['date'] == (month, day) and (event['countries'][0] == 'global' or selected_country in event['countries']):
                    st.info(f"**Special Event:** {event['name']}")
                    st.write(f"**Demand Impact:** +{int((event['demand_factor'] - 1) * 100)}%")
    
    return selected_date, selected_country

if __name__ == "__main__":
    # For testing
    st.title("Holiday Calendar Test")
    selected_date, selected_country = holiday_calendar_picker(
        title="Select a Date",
        default_date=datetime.date.today()
    )
    
    st.write(f"You selected: {selected_date} in {selected_country}")
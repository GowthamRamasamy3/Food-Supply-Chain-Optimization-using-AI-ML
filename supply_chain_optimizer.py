import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SupplyChainOptimizer:
    """
    Class for optimizing food supply chain based on demand forecasts.
    Calculates inventory requirements, costs, and provides recommendations.
    """
    
    def __init__(self, safety_buffer=0.1, lead_time=3, spoilage_rate=0.05, 
                 transport_efficiency=0.9, storage_cost=0.5, transport_cost=2.0, 
                 shortage_cost=10.0):
        """
        Initialize the supply chain optimizer.
        
        Args:
            safety_buffer (float): Safety buffer percentage (0.1 = 10%)
            lead_time (int): Lead time in days
            spoilage_rate (float): Spoilage rate percentage (0.05 = 5%)
            transport_efficiency (float): Transport efficiency percentage (0.9 = 90%)
            storage_cost (float): Storage cost per unit per day
            transport_cost (float): Transport cost per unit
            shortage_cost (float): Shortage cost per unit
        """
        self.safety_buffer = safety_buffer
        self.lead_time = lead_time
        self.spoilage_rate = spoilage_rate
        self.transport_efficiency = transport_efficiency
        self.storage_cost = storage_cost
        self.transport_cost = transport_cost
        self.shortage_cost = shortage_cost
    
    def optimize(self, forecast_df):
        """
        Run the supply chain optimization based on demand forecasts.
        
        Args:
            forecast_df (pd.DataFrame): Forecasted demand dataframe
            
        Returns:
            dict: Optimization results including inventory requirements,
                  cost breakdown, and recommendations
        """
        # Calculate inventory requirements with safety buffer
        inventory_requirements = self._calculate_inventory_requirements(forecast_df)
        
        # Calculate inventory timeline
        inventory_timeline = self._calculate_inventory_timeline(forecast_df, inventory_requirements)
        
        # Calculate costs
        cost_breakdown, total_cost = self._calculate_costs(inventory_requirements, inventory_timeline)
        
        # Calculate service level and waste
        service_level = self._calculate_service_level(inventory_timeline)
        total_waste = self._calculate_waste(inventory_requirements)
        avg_inventory = inventory_timeline['inventory_level'].mean()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            inventory_requirements, 
            inventory_timeline,
            cost_breakdown,
            service_level,
            total_waste
        )
        
        # Compile results
        results = {
            'inventory_requirements': inventory_requirements,
            'inventory_timeline': inventory_timeline,
            'cost_breakdown': cost_breakdown,
            'total_cost': total_cost,
            'service_level': service_level,
            'total_waste': total_waste,
            'avg_inventory': avg_inventory,
            'recommendations': recommendations
        }
        
        return results
    
    def _calculate_inventory_requirements(self, forecast_df):
        """
        Calculate inventory requirements with safety buffer.
        
        Args:
            forecast_df (pd.DataFrame): Forecasted demand dataframe
            
        Returns:
            pd.DataFrame: Inventory requirements
        """
        # Copy the forecast dataframe
        inventory_df = forecast_df.copy()
        
        # Calculate base inventory with safety buffer
        inventory_df['base_inventory'] = inventory_df['forecasted_demand'] * (1 + self.safety_buffer)
        
        # Account for lead time - aggregate demand by center_id for the lead time period
        inventory_df['lead_time_demand'] = inventory_df.groupby(['center_id', 'meal_id'])['forecasted_demand'].transform(
            lambda x: x.shift(-1).fillna(x).rolling(min_periods=1, window=self.lead_time).sum() / self.lead_time
        )
        
        # Calculate required inventory considering lead time
        inventory_df['required_inventory'] = np.maximum(
            inventory_df['base_inventory'],
            inventory_df['lead_time_demand'] * self.lead_time * (1 + self.safety_buffer)
        )
        
        # Account for spoilage
        inventory_df['spoilage_adjustment'] = inventory_df['required_inventory'] * self.spoilage_rate
        inventory_df['final_inventory'] = inventory_df['required_inventory'] + inventory_df['spoilage_adjustment']
        
        # Account for transport efficiency
        inventory_df['transport_adjustment'] = inventory_df['final_inventory'] * (1 / self.transport_efficiency - 1)
        inventory_df['total_inventory'] = inventory_df['final_inventory'] + inventory_df['transport_adjustment']
        
        # Round to whole units
        inventory_df['total_inventory'] = np.ceil(inventory_df['total_inventory']).astype(int)
        
        # Select relevant columns
        result_df = inventory_df[['week', 'center_id', 'meal_id', 'forecasted_demand', 'total_inventory']]
        result_df = result_df.rename(columns={'total_inventory': 'required_inventory'})
        
        return result_df
    
    def _calculate_inventory_timeline(self, forecast_df, inventory_requirements):
        """
        Calculate projected inventory levels over time.
        
        Args:
            forecast_df (pd.DataFrame): Forecasted demand dataframe
            inventory_requirements (pd.DataFrame): Calculated inventory requirements
            
        Returns:
            pd.DataFrame: Inventory timeline
        """
        # Start with a day-by-day timeline
        weeks = forecast_df['week'].unique()
        days_per_week = 7
        
        # Initialize an empty timeline
        timeline_data = []
        
        # Simulate inventory for each center and meal
        for center_id in forecast_df['center_id'].unique():
            for meal_id in forecast_df['meal_id'].unique():
                # Get requirements for this center and meal
                center_meal_reqs = inventory_requirements[
                    (inventory_requirements['center_id'] == center_id) & 
                    (inventory_requirements['meal_id'] == meal_id)
                ]
                
                # Get forecasts for this center and meal
                center_meal_forecast = forecast_df[
                    (forecast_df['center_id'] == center_id) & 
                    (forecast_df['meal_id'] == meal_id)
                ]
                
                if center_meal_reqs.empty or center_meal_forecast.empty:
                    continue
                
                # Start with initial inventory
                current_inventory = center_meal_reqs.iloc[0]['required_inventory']
                
                # Simulate day by day
                for week in weeks:
                    # Get requirements and forecast for this week
                    week_req = center_meal_reqs[center_meal_reqs['week'] == week]
                    week_forecast = center_meal_forecast[center_meal_forecast['week'] == week]
                    
                    if week_req.empty or week_forecast.empty:
                        continue
                    
                    required_inventory = week_req.iloc[0]['required_inventory']
                    daily_demand = week_forecast.iloc[0]['forecasted_demand'] / days_per_week
                    
                    # Replenish inventory at the start of the week
                    current_inventory = required_inventory
                    
                    # Simulate each day of the week
                    for day in range(days_per_week):
                        # Calculate spoilage for the day
                        daily_spoilage = current_inventory * (self.spoilage_rate / days_per_week)
                        
                        # Record the inventory level for this day
                        timeline_data.append({
                            'center_id': center_id,
                            'meal_id': meal_id,
                            'week': week,
                            'day': day + 1,
                            'day_of_year': (week - 1) * days_per_week + day + 1,
                            'inventory_level': current_inventory,
                            'daily_demand': daily_demand,
                            'daily_spoilage': daily_spoilage
                        })
                        
                        # Update inventory for the next day
                        current_inventory = max(0, current_inventory - daily_demand - daily_spoilage)
        
        # Create a dataframe from the timeline data
        timeline_df = pd.DataFrame(timeline_data)
        
        # Calculate additional metrics
        timeline_df['inventory_cost'] = timeline_df['inventory_level'] * self.storage_cost
        timeline_df['stockout'] = timeline_df['inventory_level'] < timeline_df['daily_demand']
        timeline_df['shortage_units'] = np.where(
            timeline_df['stockout'],
            np.maximum(0, timeline_df['daily_demand'] - timeline_df['inventory_level']),
            0
        )
        timeline_df['shortage_cost'] = timeline_df['shortage_units'] * self.shortage_cost
        
        return timeline_df
    
    def _calculate_costs(self, inventory_requirements, inventory_timeline):
        """
        Calculate supply chain costs.
        
        Args:
            inventory_requirements (pd.DataFrame): Inventory requirements
            inventory_timeline (pd.DataFrame): Inventory timeline
            
        Returns:
            tuple: (cost_breakdown, total_cost)
        """
        # Calculate storage costs
        total_storage_cost = inventory_timeline['inventory_cost'].sum()
        
        # Calculate transport costs
        total_transport_units = inventory_requirements['required_inventory'].sum()
        total_transport_cost = total_transport_units * self.transport_cost
        
        # Calculate shortage costs
        total_shortage_cost = inventory_timeline['shortage_cost'].sum()
        
        # Calculate spoilage costs (as lost product value)
        total_spoilage_units = inventory_timeline['daily_spoilage'].sum()
        avg_unit_cost = (self.storage_cost * 7 + self.transport_cost) # approximate unit cost
        total_spoilage_cost = total_spoilage_units * avg_unit_cost
        
        # Total cost
        total_cost = total_storage_cost + total_transport_cost + total_shortage_cost + total_spoilage_cost
        
        # Cost breakdown
        cost_breakdown = {
            'Storage Cost': round(total_storage_cost, 2),
            'Transport Cost': round(total_transport_cost, 2),
            'Shortage Cost': round(total_shortage_cost, 2),
            'Spoilage Cost': round(total_spoilage_cost, 2)
        }
        
        return cost_breakdown, total_cost
    
    def _calculate_service_level(self, inventory_timeline):
        """
        Calculate service level (percentage of demand satisfied).
        
        Args:
            inventory_timeline (pd.DataFrame): Inventory timeline
            
        Returns:
            float: Service level (0-1)
        """
        total_demand_units = inventory_timeline['daily_demand'].sum()
        shortage_units = inventory_timeline['shortage_units'].sum()
        
        if total_demand_units == 0:
            return 1.0
        
        service_level = 1 - (shortage_units / total_demand_units)
        return max(0, min(1, service_level))
    
    def _calculate_waste(self, inventory_requirements):
        """
        Calculate total waste due to spoilage.
        
        Args:
            inventory_requirements (pd.DataFrame): Inventory requirements
            
        Returns:
            float: Total waste in units
        """
        total_inventory = inventory_requirements['required_inventory'].sum()
        total_waste = total_inventory * self.spoilage_rate
        
        return total_waste
    
    def _generate_recommendations(self, inventory_requirements, inventory_timeline, 
                                 cost_breakdown, service_level, total_waste):
        """
        Generate supply chain optimization recommendations.
        
        Args:
            inventory_requirements (pd.DataFrame): Inventory requirements
            inventory_timeline (pd.DataFrame): Inventory timeline
            cost_breakdown (dict): Cost breakdown
            service_level (float): Service level
            total_waste (float): Total waste
            
        Returns:
            list: Recommendations
        """
        recommendations = []
        
        # Identify high-waste meals/centers
        meal_waste = inventory_timeline.groupby('meal_id')['daily_spoilage'].sum().reset_index()
        meal_waste = meal_waste.sort_values('daily_spoilage', ascending=False)
        
        if not meal_waste.empty:
            top_waste_meal = meal_waste.iloc[0]['meal_id']
            recommendations.append(
                f"Meal ID {top_waste_meal} has the highest spoilage. Consider adjusting production or shelf life."
            )
        
        # Identify stockout patterns
        stockout_by_day = inventory_timeline.groupby('day')['stockout'].mean().reset_index()
        stockout_by_day = stockout_by_day.sort_values('stockout', ascending=False)
        
        if not stockout_by_day.empty and stockout_by_day.iloc[0]['stockout'] > 0.1:
            high_stockout_day = stockout_by_day.iloc[0]['day']
            recommendations.append(
                f"Day {high_stockout_day} of the week shows higher stockout rates. Consider increasing delivery frequency."
            )
        
        # Service level recommendations
        if service_level < 0.95:
            recommendations.append(
                f"Current service level is {service_level:.2%}, which is below target. Consider increasing safety buffer."
            )
        elif service_level > 0.99 and self.safety_buffer > 0.05:
            recommendations.append(
                f"Service level is very high at {service_level:.2%}. Consider reducing safety buffer to optimize costs."
            )
        
        # Cost optimization recommendations
        costs = list(cost_breakdown.items())
        costs.sort(key=lambda x: x[1], reverse=True)
        highest_cost_category = costs[0][0]
        
        if highest_cost_category == 'Storage Cost':
            recommendations.append(
                "Storage costs are dominant. Consider just-in-time delivery or reducing inventory levels."
            )
        elif highest_cost_category == 'Transport Cost':
            recommendations.append(
                "Transport costs are dominant. Consider optimizing delivery routes or transport methods."
            )
        elif highest_cost_category == 'Shortage Cost':
            recommendations.append(
                "Shortage costs are significant. Increase safety buffer or improve forecast accuracy."
            )
        elif highest_cost_category == 'Spoilage Cost':
            recommendations.append(
                "Spoilage costs are high. Improve inventory rotation or consider shorter shelf-life for affected products."
            )
        
        # Add general recommendation about waste reduction
        if total_waste > 100:
            recommendations.append(
                f"Total waste is approximately {total_waste:.0f} units. Consider implementing FIFO (First In, First Out) inventory management."
            )
        
        return recommendations

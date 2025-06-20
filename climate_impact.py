import pandas as pd
import numpy as np
from datetime import datetime

class ClimateImpactAnalyzer:
    """
    Class for analyzing the climate impact of food supply chain operations.
    Calculates carbon footprint, emissions breakdowns, and sustainability score.
    """
    
    def __init__(self):
        """Initialize the climate impact analyzer with default emission factors."""
        # Emission factors (kg CO2e per unit)
        self.emission_factors = {
            # Transport emissions by mode (kg CO2e per unit per km)
            'transport': {
                'Truck': 0.1,
                'Rail': 0.03,
                'Ship': 0.02,
                'Air': 2.0
            },
            # Storage emissions (kg CO2e per unit per day)
            'storage': {
                'standard': 0.5,
                'renewable': 0.2
            },
            # Packaging emissions (kg CO2e per unit)
            'packaging': {
                'Plastic': 0.5,
                'Paper': 0.25,
                'Biodegradable': 0.15,
                'Reusable': 0.1
            },
            # Food waste emissions (kg CO2e per unit)
            'food_waste': 2.5
        }
        
        # Industry average benchmarks (for comparison)
        self.industry_benchmarks = {
            'carbon_intensity': 0.8,  # kg CO2e per $ revenue
            'waste_rate': 0.15,       # 15% waste rate
            'renewable_energy': 0.2,  # 20% renewable energy
            'transport_efficiency': 0.85  # 85% transport efficiency
        }
    
    def analyze_impact(self, forecast_df, optimization_results, transport_distance=250,
                      transport_method='Truck', renewable_energy=0.3, packaging_type='Plastic'):
        """
        Analyze the climate impact of the supply chain.
        
        Args:
            forecast_df (pd.DataFrame): Forecasted demand dataframe
            optimization_results (dict): Supply chain optimization results
            transport_distance (float): Average transport distance in kilometers
            transport_method (str): Primary transport method ('Truck', 'Rail', 'Ship', 'Air')
            renewable_energy (float): Proportion of renewable energy used (0-1)
            packaging_type (str): Primary packaging type ('Plastic', 'Paper', 'Biodegradable', 'Reusable')
            
        Returns:
            dict: Climate impact analysis results
        """
        # Extract relevant data from optimization results
        inventory_requirements = optimization_results['inventory_requirements']
        inventory_timeline = optimization_results['inventory_timeline']
        total_waste = optimization_results['total_waste']
        total_cost = optimization_results['total_cost']
        
        # Calculate total units
        total_units = inventory_requirements['required_inventory'].sum()
        total_demand = forecast_df['forecasted_demand'].sum()
        
        # Calculate waste rate
        waste_rate = total_waste / total_units if total_units > 0 else 0
        
        # Calculate emissions
        transport_emissions = self._calculate_transport_emissions(
            total_units, transport_distance, transport_method
        )
        
        storage_emissions = self._calculate_storage_emissions(
            inventory_timeline, renewable_energy
        )
        
        packaging_emissions = self._calculate_packaging_emissions(
            total_units, packaging_type
        )
        
        waste_emissions = self._calculate_waste_emissions(total_waste)
        
        # Total emissions
        total_emissions = transport_emissions + storage_emissions + packaging_emissions + waste_emissions
        
        # Emissions per unit
        emissions_per_unit = total_emissions * 1000 / total_demand if total_demand > 0 else 0
        
        # Carbon intensity (emissions per $ revenue)
        # Assuming average revenue of $10 per unit for calculation
        avg_revenue_per_unit = 10
        total_revenue = total_demand * avg_revenue_per_unit
        carbon_intensity = total_emissions * 1000 / total_revenue if total_revenue > 0 else 0
        
        # Emissions breakdown
        emissions_breakdown = {
            'Transport': round(transport_emissions, 2),
            'Storage': round(storage_emissions, 2),
            'Packaging': round(packaging_emissions, 2),
            'Food Waste': round(waste_emissions, 2)
        }
        
        # Calculate sustainability score (0-100)
        sustainability_score = self._calculate_sustainability_score(
            carbon_intensity,
            waste_rate,
            renewable_energy,
            packaging_type,
            transport_method
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            emissions_breakdown,
            carbon_intensity,
            waste_rate,
            renewable_energy,
            transport_method,
            packaging_type
        )
        
        # Create impact timeline
        impact_timeline = self._create_impact_timeline(
            inventory_timeline,
            transport_method,
            renewable_energy,
            packaging_type
        )
        
        # Compare to industry averages
        industry_comparison = self._compare_to_industry(
            carbon_intensity,
            waste_rate,
            renewable_energy,
            optimization_results['service_level']
        )
        
        # Compile results
        results = {
            'total_emissions': round(total_emissions, 2),  # tonnes CO2e
            'emissions_per_unit': round(emissions_per_unit, 2),  # kg CO2e per unit
            'carbon_intensity': round(carbon_intensity, 2),  # kg CO2e per $ revenue
            'emissions_breakdown': emissions_breakdown,
            'sustainability_score': round(sustainability_score, 1),
            'recommendations': recommendations,
            'impact_timeline': impact_timeline,
            'industry_comparison': industry_comparison
        }
        
        return results
    
    def _calculate_transport_emissions(self, total_units, transport_distance, transport_method):
        """
        Calculate emissions from transportation.
        
        Args:
            total_units (float): Total units transported
            transport_distance (float): Average transport distance in kilometers
            transport_method (str): Transport method
            
        Returns:
            float: Transport emissions in tonnes CO2e
        """
        emission_factor = self.emission_factors['transport'].get(transport_method, 0.1)
        emissions = total_units * transport_distance * emission_factor / 1000  # Convert to tonnes
        return emissions
    
    def _calculate_storage_emissions(self, inventory_timeline, renewable_energy):
        """
        Calculate emissions from storage.
        
        Args:
            inventory_timeline (pd.DataFrame): Inventory timeline
            renewable_energy (float): Proportion of renewable energy used (0-1)
            
        Returns:
            float: Storage emissions in tonnes CO2e
        """
        # Calculate total inventory-days
        total_inventory_days = inventory_timeline['inventory_level'].sum()
        
        # Calculate emissions based on energy mix
        standard_emission_factor = self.emission_factors['storage']['standard']
        renewable_emission_factor = self.emission_factors['storage']['renewable']
        
        # Weighted emission factor
        weighted_factor = (1 - renewable_energy) * standard_emission_factor + renewable_energy * renewable_emission_factor
        
        # Calculate emissions
        emissions = total_inventory_days * weighted_factor / 1000  # Convert to tonnes
        return emissions
    
    def _calculate_packaging_emissions(self, total_units, packaging_type):
        """
        Calculate emissions from packaging.
        
        Args:
            total_units (float): Total units packaged
            packaging_type (str): Type of packaging
            
        Returns:
            float: Packaging emissions in tonnes CO2e
        """
        emission_factor = self.emission_factors['packaging'].get(packaging_type, 0.5)
        emissions = total_units * emission_factor / 1000  # Convert to tonnes
        return emissions
    
    def _calculate_waste_emissions(self, total_waste):
        """
        Calculate emissions from food waste.
        
        Args:
            total_waste (float): Total food waste in units
            
        Returns:
            float: Waste emissions in tonnes CO2e
        """
        emission_factor = self.emission_factors['food_waste']
        emissions = total_waste * emission_factor / 1000  # Convert to tonnes
        return emissions
    
    def _calculate_sustainability_score(self, carbon_intensity, waste_rate, 
                                      renewable_energy, packaging_type, transport_method):
        """
        Calculate overall sustainability score (0-100).
        
        Args:
            carbon_intensity (float): Carbon intensity in kg CO2e per $ revenue
            waste_rate (float): Waste rate (0-1)
            renewable_energy (float): Proportion of renewable energy used (0-1)
            packaging_type (str): Type of packaging
            transport_method (str): Transport method
            
        Returns:
            float: Sustainability score (0-100)
        """
        # Carbon intensity score (0-30)
        # Lower is better, benchmark is 0.8 kg CO2e per $ revenue
        carbon_score = max(0, 30 * (1 - carbon_intensity / 1.5))
        
        # Waste rate score (0-25)
        # Lower is better, benchmark is 15% waste
        waste_score = max(0, 25 * (1 - waste_rate / 0.3))
        
        # Renewable energy score (0-20)
        # Higher is better
        energy_score = 20 * renewable_energy
        
        # Packaging score (0-15)
        # Based on packaging type
        packaging_scores = {
            'Plastic': 3,
            'Paper': 8,
            'Biodegradable': 12,
            'Reusable': 15
        }
        packaging_score = packaging_scores.get(packaging_type, 3)
        
        # Transport method score (0-10)
        # Based on transport method
        transport_scores = {
            'Air': 1,
            'Truck': 5,
            'Ship': 7,
            'Rail': 10
        }
        transport_score = transport_scores.get(transport_method, 5)
        
        # Total score
        total_score = carbon_score + waste_score + energy_score + packaging_score + transport_score
        
        return total_score
    
    def _generate_recommendations(self, emissions_breakdown, carbon_intensity, 
                               waste_rate, renewable_energy, transport_method, packaging_type):
        """
        Generate recommendations for improving sustainability.
        
        Args:
            emissions_breakdown (dict): Breakdown of emissions by source
            carbon_intensity (float): Carbon intensity in kg CO2e per $ revenue
            waste_rate (float): Waste rate (0-1)
            renewable_energy (float): Proportion of renewable energy used (0-1)
            transport_method (str): Transport method
            packaging_type (str): Type of packaging
            
        Returns:
            list: Sustainability recommendations
        """
        recommendations = []
        
        # Find highest emission source
        highest_source = max(emissions_breakdown.items(), key=lambda x: x[1])
        
        # Recommendations based on highest emission source
        if highest_source[0] == 'Transport':
            if transport_method == 'Air':
                recommendations.append(
                    "Switch from air freight to rail or ship transport to significantly reduce emissions."
                )
            elif transport_method == 'Truck':
                recommendations.append(
                    "Consider using rail transport for long distances or optimizing delivery routes."
                )
                
        elif highest_source[0] == 'Storage':
            if renewable_energy < 0.5:
                recommendations.append(
                    f"Increase renewable energy use from {renewable_energy*100:.0f}% to at least 50% in storage facilities."
                )
            recommendations.append(
                "Improve warehouse energy efficiency with better insulation and cold storage technologies."
            )
                
        elif highest_source[0] == 'Packaging':
            if packaging_type == 'Plastic':
                recommendations.append(
                    "Switch from plastic to biodegradable or reusable packaging to reduce emissions."
                )
            elif packaging_type == 'Paper':
                recommendations.append(
                    "Consider reusable packaging options to further reduce packaging emissions."
                )
                
        elif highest_source[0] == 'Food Waste':
            recommendations.append(
                f"Current waste rate of {waste_rate*100:.1f}% is contributing significantly to emissions. " +
                "Improve inventory management and implement better forecasting."
            )
        
        # General recommendations
        if carbon_intensity > 0.8:
            recommendations.append(
                f"Carbon intensity ({carbon_intensity:.2f} kg CO2e/$) is above industry average. " +
                "Consider setting emission reduction targets."
            )
        
        if renewable_energy < 0.3:
            recommendations.append(
                "Increase use of renewable energy across the supply chain."
            )
        
        # Add recommendation about food miles if not already covered
        if 'Transport' not in highest_source[0]:
            recommendations.append(
                "Source ingredients locally where possible to reduce food miles and transportation emissions."
            )
        
        # Add recommendation about seasonal menus if not many recommendations yet
        if len(recommendations) < 3:
            recommendations.append(
                "Design seasonal menus to reduce the climate impact of out-of-season ingredients."
            )
        
        return recommendations
    
    def _create_impact_timeline(self, inventory_timeline, transport_method, 
                             renewable_energy, packaging_type):
        """
        Create a timeline of environmental impact.
        
        Args:
            inventory_timeline (pd.DataFrame): Inventory timeline
            transport_method (str): Transport method
            renewable_energy (float): Proportion of renewable energy used (0-1)
            packaging_type (str): Type of packaging
            
        Returns:
            pd.DataFrame: Environmental impact timeline
        """
        # Create a copy of the timeline with week and day columns
        impact_df = inventory_timeline[['week', 'day', 'day_of_year']].drop_duplicates()
        
        # Group by day_of_year and get sum of inventory and spoilage
        daily_impact = inventory_timeline.groupby('day_of_year').agg({
            'inventory_level': 'sum',
            'daily_demand': 'sum',
            'daily_spoilage': 'sum'
        }).reset_index()
        
        # Merge with impact_df
        impact_df = pd.merge(impact_df, daily_impact, on='day_of_year', how='left')
        
        # Calculate daily emissions
        # Storage emissions
        storage_factor = (1 - renewable_energy) * self.emission_factors['storage']['standard'] + \
                         renewable_energy * self.emission_factors['storage']['renewable']
        impact_df['storage_emissions'] = impact_df['inventory_level'] * storage_factor
        
        # Transport emissions (assuming daily replenishment)
        transport_factor = self.emission_factors['transport'].get(transport_method, 0.1)
        avg_transport_distance = 50  # Assuming 50km average daily transport distance
        impact_df['transport_emissions'] = impact_df['daily_demand'] * transport_factor * avg_transport_distance
        
        # Packaging emissions
        packaging_factor = self.emission_factors['packaging'].get(packaging_type, 0.5)
        impact_df['packaging_emissions'] = impact_df['daily_demand'] * packaging_factor
        
        # Waste emissions
        waste_factor = self.emission_factors['food_waste']
        impact_df['waste_emissions'] = impact_df['daily_spoilage'] * waste_factor
        
        # Total daily emissions
        impact_df['total_emissions'] = impact_df['storage_emissions'] + \
                                      impact_df['transport_emissions'] + \
                                      impact_df['packaging_emissions'] + \
                                      impact_df['waste_emissions']
        
        # Calculate cumulative emissions
        impact_df['cumulative_emissions'] = impact_df['total_emissions'].cumsum()
        
        # Calculate a simple daily environmental risk index (0-100, higher is more risk)
        max_daily_emissions = impact_df['total_emissions'].max()
        if max_daily_emissions > 0:
            impact_df['environmental_risk_index'] = 100 * impact_df['total_emissions'] / max_daily_emissions
        else:
            impact_df['environmental_risk_index'] = 0
            
        # Sort by day of year
        impact_df = impact_df.sort_values('day_of_year')
        
        return impact_df
    
    def _compare_to_industry(self, carbon_intensity, waste_rate, renewable_energy, service_level):
        """
        Compare sustainability metrics to industry benchmarks.
        
        Args:
            carbon_intensity (float): Carbon intensity in kg CO2e per $ revenue
            waste_rate (float): Waste rate (0-1)
            renewable_energy (float): Proportion of renewable energy used (0-1)
            service_level (float): Service level (0-1)
            
        Returns:
            dict: Comparison to industry benchmarks (percentage difference)
        """
        # Calculate percentage differences
        # For carbon_intensity and waste_rate, lower is better (negative % difference is good)
        # For renewable_energy and service_level, higher is better (positive % difference is good)
        
        carbon_diff = (self.industry_benchmarks['carbon_intensity'] - carbon_intensity) / \
                      self.industry_benchmarks['carbon_intensity'] * 100
        
        waste_diff = (self.industry_benchmarks['waste_rate'] - waste_rate) / \
                    self.industry_benchmarks['waste_rate'] * 100
        
        renewable_diff = (renewable_energy - self.industry_benchmarks['renewable_energy']) / \
                        self.industry_benchmarks['renewable_energy'] * 100
        
        # Assume industry average service level of 95%
        industry_service_level = 0.95
        service_diff = (service_level - industry_service_level) / industry_service_level * 100
        
        comparison = {
            'Carbon Intensity': round(carbon_diff, 1),
            'Waste Rate': round(waste_diff, 1),
            'Renewable Energy': round(renewable_diff, 1),
            'Service Level': round(service_diff, 1)
        }
        
        return comparison

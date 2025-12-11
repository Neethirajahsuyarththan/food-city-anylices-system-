import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import calendar
import os
from dataclasses import dataclass
import numpy as np
from matplotlib.ticker import FuncFormatter


# =============================================
# Data Models and Base Classes
# =============================================

@dataclass
class SaleRecord:
    branch_id: str
    branch_name: str
    date: str
    product_id: str
    product_name: str
    sale_quantity: int
    price_per_one: float
    total_price: float


@dataclass
class StockRecord:
    branch_id: str
    branch_name: str
    date: str
    product_id: str
    product_name: str
    product_quantity: int
    price_per_one: float


class DataLoader(ABC):
    @abstractmethod
    def load_sales_data(self) -> List[SaleRecord]:
        pass

    @abstractmethod
    def load_stock_data(self) -> List[StockRecord]:
        pass


class CSVDataLoader(DataLoader):
    def __init__(self, sales_file: str, stock_file: str):
        self.sales_file = sales_file
        self.stock_file = stock_file

    def load_sales_data(self) -> List[SaleRecord]:
        try:
            df = pd.read_csv(self.sales_file)
            records = []
            for _, row in df.iterrows():
                records.append(SaleRecord(
                    branch_id=row['Branch ID'],
                    branch_name=row['Branch Name'],
                    date=row['Date'],
                    product_id=row['Product ID'],
                    product_name=row['Product Name'],
                    sale_quantity=row['Sale Quantity'],
                    price_per_one=row['Price per One'],
                    total_price=row['Total Price']
                ))
            return records
        except Exception as e:
            raise Exception(f"Error loading sales data: {str(e)}")

    def load_stock_data(self) -> List[StockRecord]:
        try:
            df = pd.read_csv(self.stock_file)
            records = []
            for _, row in df.iterrows():
                records.append(StockRecord(
                    branch_id=row['Branch ID'],
                    branch_name=row['Branch Name'],
                    date=row['Date'],
                    product_id=row['Product ID'],
                    product_name=row['Product Name'],
                    product_quantity=row['Product Quantity'],
                    price_per_one=row['Price Per One']
                ))
            return records
        except Exception as e:
            raise Exception(f"Error loading stock data: {str(e)}")


# =============================================
# Analysis Strategies
# =============================================

class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, sales_data: List[SaleRecord], stock_data: List[StockRecord], params: dict):
        pass


class MonthlyBranchSalesAnalysis(AnalysisStrategy):
    def analyze(self, sales_data: List[SaleRecord], stock_data: List[StockRecord], params: dict):
        month = params.get('month')
        year = params.get('year')

        if not month or not year:
            raise ValueError("Month and year are required parameters")

        # Convert sales data to DataFrame for easier manipulation
        df = pd.DataFrame([{
            'branch_id': sale.branch_id,
            'branch_name': sale.branch_name,
            'date': sale.date,
            'total_price': sale.total_price
        } for sale in sales_data])

        # Convert date strings to datetime objects with dayfirst=True for consistent parsing
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

        # Filter by month and year
        filtered = df[(df['date'].dt.month == month) & (df['date'].dt.year == year)].copy()

        if filtered.empty:
            return None, None

        # Group by branch and sum total sales
        result = filtered.groupby(['branch_id', 'branch_name'])['total_price'].sum().reset_index()

        # Create plot
        plt.figure(figsize=(12, 7))
        bars = plt.bar(result['branch_name'], result['total_price'], color=plt.cm.Paired(np.arange(len(result))))

        # Format y-axis as currency
        formatter = FuncFormatter(lambda x, p: f"LKR {x:,.0f}")
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.title(f"Monthly Sales Analysis for {calendar.month_name[month]} {year}", pad=20, fontsize=14,
                  fontweight='bold')
        plt.xlabel("Branch", labelpad=10, fontsize=12)
        plt.ylabel("Total Sales (LKR)", labelpad=10, fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'LKR {height:,.0f}',
                     ha='center', va='bottom', fontsize=9)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        return result, plt


class ProductPriceAnalysis(AnalysisStrategy):
    def analyze(self, sales_data: List[SaleRecord], stock_data: List[StockRecord], params: dict):
        product_name = params.get('product_name')
        start_month = params.get('start_month')
        start_year = params.get('start_year')
        end_month = params.get('end_month')
        end_year = params.get('end_year')

        if not product_name:
            raise ValueError("Product name is required")

        # Create DataFrame from sales data
        sales_df = pd.DataFrame([{
            'product_name': sale.product_name,
            'date': sale.date,
            'price': sale.price_per_one
        } for sale in sales_data if sale.product_name.lower() == product_name.lower()])

        if sales_df.empty:
            return None, None

        # Convert dates with dayfirst=True for consistent parsing
        sales_df['date'] = pd.to_datetime(sales_df['date'], dayfirst=True, errors='coerce')
        sales_df = sales_df.dropna(subset=['date']).copy()

        # Filter by date range if provided
        if start_month and start_year and end_month and end_year:
            start_date = datetime(start_year, start_month, 1)
            end_date = datetime(end_year, end_month, calendar.monthrange(end_year, end_month)[1])
            sales_df = sales_df[(sales_df['date'] >= start_date) & (sales_df['date'] <= end_date)].copy()

        if sales_df.empty:
            return None, None

        # Extract month and year for grouping
        sales_df['month_year'] = sales_df['date'].dt.to_period('M')

        # Group by month and calculate average price
        monthly_avg = sales_df.groupby('month_year')['price'].mean().reset_index()
        monthly_avg['month_year'] = monthly_avg['month_year'].dt.strftime('%b %Y')
        monthly_avg = monthly_avg.sort_values('month_year')

        # Create plot
        plt.figure(figsize=(14, 7))

        # Format y-axis as currency
        formatter = FuncFormatter(lambda x, p: f"LKR {x:,.2f}")
        plt.gca().yaxis.set_major_formatter(formatter)

        # Create line plot with markers
        plt.plot(monthly_avg['month_year'], monthly_avg['price'],
                 marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=8)

        # Add data labels
        for x, y in zip(monthly_avg['month_year'], monthly_avg['price']):
            plt.text(x, y, f'LKR {y:,.2f}',
                     ha='center', va='bottom', fontsize=9, color='darkblue')

        # Set title and labels
        title = f"Monthly Price Analysis for {product_name.title()}"
        if start_month and start_year and end_month and end_year:
            title += f"\n{datetime(start_year, start_month, 1).strftime('%b %Y')} to {datetime(end_year, end_month, 1).strftime('%b %Y')}"

        plt.title(title, pad=20, fontsize=14, fontweight='bold')
        plt.xlabel("Month", labelpad=10, fontsize=12)
        plt.ylabel(" Price (LKR)", labelpad=10, fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Add grid and adjust layout
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        return monthly_avg, plt


class ProductPreferenceAnalysis(AnalysisStrategy):
    def analyze(self, sales_data: List[SaleRecord], stock_data: List[StockRecord], params: dict):
        month = params.get('month')
        year = params.get('year')
        branch_name = params.get('branch_name')

        if not month or not year:
            raise ValueError("Month and year are required parameters")

        # Convert sales data to DataFrame
        df = pd.DataFrame([{
            'branch_name': sale.branch_name,
            'date': sale.date,
            'product_name': sale.product_name,
            'sale_quantity': sale.sale_quantity
        } for sale in sales_data])

        # Convert dates with dayfirst=True for consistent parsing
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date']).copy()

        # Filter by month and year
        filtered = df[(df['date'].dt.month == month) & (df['date'].dt.year == year)].copy()

        # Filter by branch if provided
        if branch_name:
            filtered = filtered[filtered['branch_name'].str.lower() == branch_name.lower()].copy()

        if filtered.empty:
            return None, None

        # Group by product and sum quantities
        result = filtered.groupby('product_name')['sale_quantity'].sum().reset_index()
        result = result.sort_values('sale_quantity', ascending=False)

        # Create plot
        plt.figure(figsize=(14, 7))

        # Limit to top 20 products if there are many
        if len(result) > 20:
            result = result.head(20)
            title_note = " (Top 20 Products)"
        else:
            title_note = ""

        # Create horizontal bar chart for better readability
        bars = plt.barh(result['product_name'], result['sale_quantity'],
                        color=plt.cm.plasma(np.linspace(0.1, 0.9, len(result))))

        title = f"Product Preference Analysis for {calendar.month_name[month]} {year}"
        if branch_name:
            title += f" - {branch_name}"
        title += title_note

        plt.title(title, pad=20, fontsize=14, fontweight='bold')
        plt.xlabel("Total Units Sold", labelpad=10, fontsize=12)
        plt.ylabel("Product Name", labelpad=10, fontsize=12)

        # Add value labels to the right of bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + max(result['sale_quantity']) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{int(width):,}',
                     va='center', fontsize=9)

        plt.gca().invert_yaxis()  # Highest values at top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        return result, plt


class SalesDistributionAnalysis(AnalysisStrategy):
    def analyze(self, sales_data: List[SaleRecord], stock_data: List[StockRecord], params: dict):
        branch_name = params.get('branch_name')
        start_month = params.get('start_month')
        start_year = params.get('start_year')
        end_month = params.get('end_month')
        end_year = params.get('end_year')

        if not all([start_month, start_year, end_month, end_year]):
            raise ValueError("Start and end month/year are required")

        # Convert sales data to DataFrame
        sales_df = pd.DataFrame([{
            'branch_name': sale.branch_name,
            'date': sale.date,
            'product_name': sale.product_name,
            'sale_quantity': sale.sale_quantity,
            'total_price': sale.total_price
        } for sale in sales_data])

        # Convert stock data to DataFrame
        stock_df = pd.DataFrame([{
            'branch_name': stock.branch_name,
            'date': stock.date,
            'product_name': stock.product_name,
            'stock_quantity': stock.product_quantity
        } for stock in stock_data])

        # Convert dates with dayfirst=True for consistent parsing
        sales_df['date'] = pd.to_datetime(sales_df['date'], dayfirst=True, errors='coerce')
        stock_df['date'] = pd.to_datetime(stock_df['date'], dayfirst=True, errors='coerce')

        # Filter by date range
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, calendar.monthrange(end_year, end_month)[1])

        sales_filtered = sales_df[(sales_df['date'] >= start_date) & (sales_df['date'] <= end_date)].copy()
        stock_filtered = stock_df[(stock_df['date'] >= start_date) & (stock_df['date'] <= end_date)].copy()

        # Filter by branch if provided
        if branch_name:
            sales_filtered = sales_filtered[sales_filtered['branch_name'].str.lower() == branch_name.lower()].copy()
            stock_filtered = stock_filtered[stock_filtered['branch_name'].str.lower() == branch_name.lower()].copy()

        if sales_filtered.empty or stock_filtered.empty:
            return None, None

        # Group sales by product
        sales_grouped = sales_filtered.groupby('product_name').agg({
            'sale_quantity': 'sum',
            'total_price': 'sum'
        }).reset_index()

        # Group stock by product
        stock_grouped = stock_filtered.groupby('product_name')['stock_quantity'].sum().reset_index()

        # Merge sales and stock data
        merged = pd.merge(sales_grouped, stock_grouped, on='product_name', how='inner')

        # Calculate percentage of sales to stock
        merged['percentage'] = (merged['sale_quantity'] / merged['stock_quantity']) * 100

        # Sort by percentage
        merged = merged.sort_values('percentage', ascending=False)

        # Limit to top 20 products if there are many
        if len(merged) > 20:
            merged = merged.head(20)
            title_note = " (Top 20 Products)"
        else:
            title_note = ""

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 7))

        # Create a colormap based on percentage values
        colors = plt.cm.coolwarm(merged['percentage'] / 100)

        bars = ax.bar(merged['product_name'], merged['percentage'], color=colors)

        title = f"Sales Distribution Analysis\n{start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')}"
        if branch_name:
            title += f" - {branch_name}"
        title += title_note

        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
        ax.set_xlabel("Product Name", labelpad=10, fontsize=12)
        ax.set_ylabel("Percentage of Sales to Stock (%)", labelpad=10, fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)

        # Add a colorbar to show the scale
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Sales Percentage', rotation=270, labelpad=15)

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        return merged, plt


# =============================================
# Analysis Context and Factory
# =============================================

class AnalysisContext:
    def __init__(self, strategy: AnalysisStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> AnalysisStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: AnalysisStrategy) -> None:
        self._strategy = strategy

    def execute_analysis(self, sales_data: List[SaleRecord], stock_data: List[StockRecord], params: dict):
        return self._strategy.analyze(sales_data, stock_data, params)


class AnalysisFactory:
    @staticmethod
    def create_analysis(analysis_type: str) -> AnalysisStrategy:
        if analysis_type == "monthly_branch_sales":
            return MonthlyBranchSalesAnalysis()
        elif analysis_type == "product_price":
            return ProductPriceAnalysis()
        elif analysis_type == "weekly_sales":
            return WeeklySalesAnalysis()
        elif analysis_type == "product_preference":
            return ProductPreferenceAnalysis()
        elif analysis_type == "sales_distribution":
            return SalesDistributionAnalysis()
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")


# =============================================
# User Interface
# =============================================

class ConsoleUI:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.sales_data = []
        self.stock_data = []

    def load_data(self):
        try:
            print("Loading data...")
            self.sales_data = self.data_loader.load_sales_data()
            self.stock_data = self.data_loader.load_stock_data()
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {str(e)}")

    def display_menu(self):
        print("\nFood City Sales Analysis System")
        print("1. Monthly Sales Analysis by Branch")
        print("2. Product Price Analysis")
        print("3. Weekly Sales Analysis")
        print("4. Product Preference Analysis")
        print("5. Sales Distribution Analysis")
        print("6. Exit")

    def get_month_year_input(self, prompt: str) -> Tuple[int, int]:
        while True:
            try:
                date_str = input(prompt + " (MM/YYYY): ")
                month, year = map(int, date_str.split('/'))
                if month < 1 or month > 12:
                    print("Month must be between 1 and 12")
                    continue
                return month, year
            except ValueError:
                print("Invalid format. Please use MM/YYYY format.")

    def get_month_range_input(self) -> Tuple[int, int, int, int]:
        print("\nEnter start date:")
        start_month, start_year = self.get_month_year_input("Start date")
        print("\nEnter end date:")
        end_month, end_year = self.get_month_year_input("End date")
        return start_month, start_year, end_month, end_year

    def run(self):
        self.load_data()

        while True:
            self.display_menu()
            choice = input("Enter your choice (1-6): ")

            if choice == "1":
                self.handle_monthly_branch_sales()
            elif choice == "2":
                self.handle_product_price_analysis()
            elif choice == "3":
                self.handle_weekly_sales_analysis()
            elif choice == "4":
                self.handle_product_preference_analysis()
            elif choice == "5":
                self.handle_sales_distribution_analysis()
            elif choice == "6":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

    def handle_monthly_branch_sales(self):
        print("\nMonthly Sales Analysis by Branch")
        month, year = self.get_month_year_input("Enter month and year to analyze")

        context = AnalysisContext(AnalysisFactory.create_analysis("monthly_branch_sales"))
        result, plot = context.execute_analysis(
            self.sales_data,
            self.stock_data,
            {'month': month, 'year': year}
        )

        if result is None:
            print("No data available for the selected period.")
            return

        print("\nMonthly Sales by Branch:")
        print(result.to_string(index=False))

        plot.show()
        input("Press Enter to continue...")

    def handle_product_price_analysis(self):
        print("\nProduct Price Analysis")
        product_name = input("Enter product name: ").strip()

        print("\nAnalyze for:")
        print("1. Specific month/year")
        print("2. Date range")
        period_choice = input("Enter choice (1-2): ")

        params = {'product_name': product_name}

        if period_choice == "1":
            month, year = self.get_month_year_input("Enter month and year to analyze")
            params.update({
                'start_month': month,
                'start_year': year,
                'end_month': month,
                'end_year': year
            })
        elif period_choice == "2":
            start_month, start_year, end_month, end_year = self.get_month_range_input()
            params.update({
                'start_month': start_month,
                'start_year': start_year,
                'end_month': end_month,
                'end_year': end_year
            })
        else:
            print("Invalid choice")
            return

        context = AnalysisContext(AnalysisFactory.create_analysis("product_price"))
        result, plot = context.execute_analysis(
            self.sales_data,
            self.stock_data,
            params
        )

        if result is None:
            print("No data available for the selected product/period.")
            return

        print("\nPrice Analysis Results:")
        print(result.to_string(index=False))

        plot.show()
        input("Press Enter to continue...")

    def handle_weekly_sales_analysis(self):
        print("\nWeekly Sales Analysis")
        start_month, start_year, end_month, end_year = self.get_month_range_input()

        context = AnalysisContext(AnalysisFactory.create_analysis("weekly_sales"))
        result, plot = context.execute_analysis(
            self.sales_data,
            self.stock_data,
            {
                'start_month': start_month,
                'start_year': start_year,
                'end_month': end_month,
                'end_year': end_year
            }
        )

        if result is None:
            print("No data available for the selected period.")
            return

        print("\nWeekly Sales Analysis Results:")
        print(result.to_string(index=False))

        plot.show()
        input("Press Enter to continue...")

    def handle_product_preference_analysis(self):
        print("\nProduct Preference Analysis")
        month, year = self.get_month_year_input("Enter month and year to analyze")

        branch_name = input("Enter branch name (leave blank for all branches): ").strip()

        params = {
            'month': month,
            'year': year
        }

        if branch_name:
            params['branch_name'] = branch_name

        context = AnalysisContext(AnalysisFactory.create_analysis("product_preference"))
        result, plot = context.execute_analysis(
            self.sales_data,
            self.stock_data,
            params
        )

        if result is None:
            print("No data available for the selected period/branch.")
            return

        print("\nProduct Preference Results:")
        print(result.to_string(index=False))

        plot.show()
        input("Press Enter to continue...")

    def handle_sales_distribution_analysis(self):
        print("\nSales Distribution Analysis")
        start_month, start_year, end_month, end_year = self.get_month_range_input()

        branch_name = input("Enter branch name (leave blank for all branches): ").strip()

        params = {
            'start_month': start_month,
            'start_year': start_year,
            'end_month': end_month,
            'end_year': end_year
        }

        if branch_name:
            params['branch_name'] = branch_name

        context = AnalysisContext(AnalysisFactory.create_analysis("sales_distribution"))
        result, plot = context.execute_analysis(
            self.sales_data,
            self.stock_data,
            params
        )

        if result is None:
            print("No data available for the selected period/branch.")
            return

        print("\nSales Distribution Results:")
        print(result.to_string(index=False))

        plot.show()
        input("Press Enter to continue...")


# =============================================
# Main Application
# =============================================

def main():
    # Initialize data loader with the provided CSV files
    sales_file = "food_city_sales_data_2024.csv"
    stock_file = "food_city_stock_2024.csv"

    # Check if files exist
    if not os.path.exists(sales_file) or not os.path.exists(stock_file):
        print("Error: Required data files not found.")
        print(f"Please ensure both {sales_file} and {stock_file} exist in the current directory.")
        return

    # Create data loader and UI
    data_loader = CSVDataLoader(sales_file, stock_file)
    ui = ConsoleUI(data_loader)

    # Run the application
    ui.run()


if __name__ == "__main__":
    main()
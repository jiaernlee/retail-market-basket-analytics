import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import random

st.title("Retail Market Basket Analytics: Multi-Store Customer Purchase Pattern Analysis")
st.subheader("PSW Level 2 Group Challenge")

st.markdown("""
### Objective
As a team, develop a comprehensive market basket analysis solution for a major supermarket chain to optimize store layouts, improve cross-selling, and increase average transaction value by 10%. Your team will analyze transaction patterns across multiple stores and provide actionable recommendations for inventory management and promotional strategies.
""")

def extract_frozenset_items(zz):
    items = []
    for z in zz:
        items.append(list(z))
    return list(set([a for item in items for a in item]))

def determine_promotion_type(confidence):
    if confidence > 0.75:
        return "Bundle Deal"
    elif confidence > 0.65:
        return "Buy-One-Get-One-Free"
    else:
        return "10% Off"

stores_data = pd.read_csv("stores.csv")

stores_unique = stores_data['store_id'].unique()

with st.form('main'):
    store_id = st.selectbox(
        "Select a store to analyze",
        stores_unique
    )

    submit = st.form_submit_button()

if submit:
    products = pd.read_csv('products.csv')
    stores = pd.read_csv('stores.csv')
    transactions = pd.read_csv('transactions.csv')
    
    transactions = transactions[transactions['store_id'] == store_id]

    transactions['date'] = pd.to_datetime(transactions['date'], dayfirst=True)
    transactions['total_price'] = transactions['quantity'] * transactions['unit_price']
    
    store_info = transactions.copy()
    store_info = store_info.merge(stores, on='store_id').merge(products, on='category')
    store_info['is_weekend'] = store_info['date'].dt.dayofweek > 4
    store_info['hour'] = store_info['date'].dt.hour
    store_info['day'] = store_info['date'].dt.weekday
    store_info['season'] = ["Winter" if month < 3 else "Spring" if month < 6 else "Summer" if month < 9 else "Autumn" if month < 12 else "Winter" for month in store_info['date'].dt.month]

    basket = store_info.groupby(['transaction_id', 'category'])['quantity'].sum().unstack()
    basket = (basket > 0).astype(int)
    
    frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, num_itemsets=len(transactions), metric="lift", min_threshold=1.0)
        
    rules['zhangs_metric'] = (rules['confidence'] - rules['consequent support']) / (1 - rules['consequent support'])

    high_lift_rules = rules[rules['lift'] > 1.2].sort_values(by='lift',ascending=False)
    st.write("#### High-Lift Rules", high_lift_rules)

    st.write("## Market Layout Suggestions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("#### Entrance Zone")
        entrance_zone = list(set(high_lift_rules.sort_values(by="support", ascending=False).iloc[0,:2].values.flatten()))
        entrance_zone = extract_frozenset_items(entrance_zone)
        for i in range(len(entrance_zone)):
            st.write(f"{1+i}. {entrance_zone[i]}")
                
    with col2:
        st.write("#### Cross-selling Zone")
        cross_selling = list(set(high_lift_rules.sort_values(by="confidence", ascending=False).iloc[0:3,:2].values.flatten()))
        cross_selling = extract_frozenset_items(cross_selling)
        for i in range(len(cross_selling)):
            st.write(f"{i+1}. {cross_selling[i]}")
        
    with col3:
        st.write("#### Traffic Flow Zone")
        traffic_flow = list(set(high_lift_rules.sort_values(by="lift", ascending=False).iloc[0:3,:2].values.flatten()))
        traffic_flow = extract_frozenset_items(traffic_flow)
        for i in range(len(traffic_flow)):
            st.write(f"{i+1}. {traffic_flow[i]}")

    
    st.write("## Promotional Calendar")

    st.write("### Weekly Promotions")
    
    selected_day = st.selectbox(
        "Select a day",
        ['Weekdays', 'Weekends'],
        index=0
    )

    day_to_is_weekend = {
        "Weekdays": False,
        "Weekends": True,
    }

    weekly_promotions = {}

    for is_weekend in [True, False]:
        weekly_basket = store_info[store_info['is_weekend'] == is_weekend].groupby(['transaction_id', 'category'])['quantity'].sum().unstack()
        weekly_basket = (weekly_basket > 0).astype(int)
        
        weekly_frequent_itemsets = apriori(weekly_basket, min_support=0.05, use_colnames=True)
        
        weekly_rules = association_rules(weekly_frequent_itemsets, len(store_info[store_info['is_weekend'] == is_weekend]), metric="lift", min_threshold=1.0)
        weekly_rules = weekly_rules[weekly_rules['lift'] > 1.2].sort_values(by='lift',ascending=False)
        
        top_rules = weekly_rules.sort_values(by=['confidence', 'support', 'lift'], ascending=False)

        choices = [random.randint(0, len(top_rules)) for i in range(10)]
        methods = []
        for choice in choices:
            selected_row = top_rules.iloc[choice,:]
            promotional_method = determine_promotion_type(selected_row['confidence'])
            methods.append(f"{promotional_method} on {' and '.join(list(selected_row['antecedents']))} when buying {' and '.join(list(selected_row['consequents']))}")
    
        weekly_promotions[is_weekend] = methods

    for i in range(len(weekly_promotions[day_to_is_weekend[selected_day]])):
        st.write(f"{i+1}. {weekly_promotions[day_to_is_weekend[selected_day]][i]}")

    st.write("### Seasonal Promotions")

    selected_season = st.selectbox(
        'Select a season',
        ['Spring', 'Summer', 'Autumn', "Winter"],
        index=0
    )

    seasonal_promotions = {}

    for season in ['Spring', 'Summer', 'Autumn', "Winter"]:
        seasonal_basket = store_info[store_info['season'] == season].groupby(['transaction_id', 'category'])['quantity'].sum().unstack()
        seasonal_basket = (seasonal_basket > 0).astype(int)
        
        seasonal_frequent_itemsets = apriori(seasonal_basket, min_support=0.05, use_colnames=True)
        
        seasonal_rules = association_rules(seasonal_frequent_itemsets, len(store_info[store_info['season'] == season]), metric="lift", min_threshold=1.0)
        seasonal_rules = seasonal_rules[seasonal_rules['lift'] > 1.2].sort_values(by='lift',ascending=False)
        
        top_rules = seasonal_rules.sort_values(by=['confidence', 'support', 'lift'], ascending=False)

        choices = [random.randint(0, len(top_rules)) for i in range(10)]
        methods = []
        for choice in choices:
            selected_row = top_rules.iloc[choice,:]
            promotional_method = determine_promotion_type(selected_row['confidence'])
            methods.append(f"{promotional_method} on {' and '.join(list(selected_row['antecedents']))} when buying {' and '.join(list(selected_row['consequents']))}")
    
        seasonal_promotions[season] = methods

    for i in range(len(seasonal_promotions[selected_season])):
        st.write(f"{i+1}. {seasonal_promotions[selected_season][i]}")
    
    
    
    
    
    
    
    
    






    
    

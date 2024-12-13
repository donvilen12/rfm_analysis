import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from datetime import datetime
from datetime import timedelta


sns.set(style='dark')

# Helper functions for various dataframe creation

def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_approved_at').agg({
        "order_id": "nunique",
        "payment_value": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)
    
    return daily_orders_df


def create_sum_spend_df(df):
    sum_spend_df = df.resample(rule='D', on='order_approved_at').agg({
        "payment_value": "sum"
    })
    sum_spend_df = sum_spend_df.reset_index()
    sum_spend_df.rename(columns={
        "payment_value": "total_spend"
    }, inplace=True)

    return sum_spend_df

def create_sum_order_items_df(df):
    sum_order_items_df = df.groupby("product_category_name_english")["product_id"].count().reset_index()
    sum_order_items_df.rename(columns={
        "product_id": "product_count"
    }, inplace=True)
    sum_order_items_df = sum_order_items_df.sort_values(by='product_count', ascending=False)

    return sum_order_items_df


def review_score_df(df):
    review_scores = df['review_score'].value_counts().sort_values(ascending=False)
    most_common_score = review_scores.idxmax()

    return review_scores, most_common_score


def create_sales_long_format(df):
    monthly_order = df.resample(rule='M', on='order_approved_at').agg({
        "price": "sum",
    })

    # convert an index of a dataframe into a column
    monthly_order = monthly_order.reset_index()

    monthly_order['order_purchase_year'] = monthly_order['order_approved_at'].apply(lambda x: x.year)
    monthly_order['order_purchase_month'] = monthly_order['order_approved_at'].apply(lambda x: x.month)

    # select only sales in 2017 & 2018
    df_orders_compare = monthly_order.query('order_purchase_year in (2017, 2018) & order_purchase_month <= 12')

    sales_by_month = df_orders_compare.groupby(['order_purchase_year', 'order_purchase_month'])['price'].sum().unstack()

    sales_long_format = sales_by_month.reset_index().melt(id_vars='order_purchase_year', var_name='order_purchase_month', value_name='total_sales')

    return sales_long_format


# Helper functions for RFM #

# Define RFM segments based on RFM scores
def segment_rfm(score):
    if score == '333':
        return 'champion'
    elif score in ('332|331|323|313'):
        return 'potential1'
    elif score in ('321|322|311|312'):
        return 'potential2'
    elif score in '233' :
        return 'needing_attention1'
    elif score in ('223|213|212|231|232|211|221|222'):
        return 'needing_attention2'
    elif score in ('132|123|113|133'):
        return 'lost1'
    elif score in ('111|112|121|122|131') :
        return 'lost2'
    else:
        return 'no label'

def create_rfm_df(df):
    # Group by CustomerID to calculate RFM
    rfm = df.groupby('customer_id').agg(
    max_order_timestamp=("order_purchase_timestamp", "max"),
    Frequency=("order_id", "count"),
    Monetary=("price", "sum")
    ).reset_index()

    # Calcuate the last date customers do the transaction in days
    rfm["max_order_timestamp"] = rfm["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max() + timedelta(days=1)
    rfm["Recency"] = rfm["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    
    # Calculating quantiles values
    quintiles = rfm[['Recency', 'Frequency', 'Monetary']].quantile([.2, .25, .3, .35, .4, .5, .6, .7, .8, .9]).to_dict()

    # Benchmark to give score for recency indicator
    def r_score(r):
        if r < quintiles['Recency'][.2]:
            return 3 
        elif r < quintiles['Recency'][.8]:
            return 2
        else: 
            return 1

    # Benchmark to give score for frequency & monetary indicator.   
    def fm_score(f): 
        if f > quintiles['Frequency'][.8]:
            return 3
        elif f > quintiles['Frequency'][.2]: 
            return 2
        else: 
            return 1
    
    rfm['R_score'] = rfm.Recency.apply(lambda x: r_score(x))
    rfm['F_score'] = rfm.Frequency.apply(lambda x: fm_score(x))
    rfm['M_score'] = rfm.Monetary.apply(lambda x: fm_score(x))
    rfm['RFM_Score'] = rfm['R_score'].map(str)+rfm['F_score'].map(str) + rfm['M_score'].map(str)

    # Mapping Label based on score
    rfm['Label'] = rfm['RFM_Score'].apply(segment_rfm)

    # Calculate Percentiles
    rfm_percentiles = rfm[['Recency', 'Frequency', 'Monetary']].rank(pct=True)

    # RFM Segmentation
    rfm['Percent_Score'] = rfm_percentiles['Recency'] * 100 + rfm_percentiles['Frequency'] * 10 + rfm_percentiles['Monetary']
    rfm['Percent_Score'] = rfm['Percent_Score'].clip(0, 100)
    rfm_segment = pd.cut(rfm['Percent_Score'], bins=[0, 33, 66, 100], labels=['Low', 'Mid', 'High'])
    rfm['Segment'] = rfm_segment

    return rfm


# Load cleaned data
all_df = pd.read_csv("transactions.csv")

datetime_columns = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp", "shipping_limit_date"]
all_df.sort_values(by="order_approved_at", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Filter data
min_date = all_df["order_approved_at"].min()
max_date = all_df["order_approved_at"].max()

with st.sidebar:
    # Display logo 
    st.sidebar.image(load_image("olist.png"), use_column_width=True)
    # st.image("https://github.com/donvilen12/rfm_analysis/blob/main/olist.PNG")
    # st.image("https://github.com/donvilen12/rfm_analysis/blob/main/braz-eComS1.PNG")
    st.write("#")
    
    # Get start_date & end_date from date_input
    start_date, end_date = st.date_input(
        label='Duration',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df["order_approved_at"] >= str(start_date)) & 
                (all_df["order_approved_at"] <= str(end_date))]

# st.dataframe(main_df)

# Prepare various dataframe
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
sum_spend_df = create_sum_spend_df(main_df)
review_score, common_score = review_score_df(main_df)
sales_long_format_df = create_sales_long_format(main_df)
rfm_df = create_rfm_df(main_df)

st.header(':sparkles: Brazilian E-Commerce Dashboard :sparkles:')

# Plot Daily Orders
st.subheader('Daily Orders')
col1, col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total orders", value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "R$", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_approved_at"],
    daily_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#425a90"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

# Setting font dictionary  
font = {'family': 'Verdana', 
        'color':  'black', 
        'size': 20, 
        } 

plt.ylabel("# order", fontdict = font)

st.pyplot(fig)

 

# Customer Spend Money -> Revenue
st.subheader("Revenue")
col1, col2 = st.columns(2)

with col1:
    total_spend = format_currency(sum_spend_df["total_spend"].sum(), "R$", locale="es_CO")
    st.markdown(f"Total Revenue: **{total_spend}**")

with col2:
    avg_spend = format_currency(sum_spend_df["total_spend"].mean(), "R$", locale="es_CO")
    st.markdown(f"Average Revenue: **{avg_spend}**")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    sum_spend_df["order_approved_at"],
    sum_spend_df["total_spend"],
    marker="o",
    linewidth=2,
    color="#425a90"
)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", labelsize=15)
st.pyplot(fig)

# Product performance
st.subheader("Best & Worst Performing Product")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="product_count", y="product_category_name_english", data=sum_order_items_df.head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Number of Sales", fontsize=30)
ax[0].set_title("Best Performing Product", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)

sns.barplot(x="product_count", y="product_category_name_english", data=sum_order_items_df.sort_values(by="product_count", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Number of Sales", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# Review Score
st.subheader("Review Score")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=review_score.index, 
            y=review_score.values, 
            order=review_score.index,
            palette=["#90CAF9" if score == common_score else "#D3D3D3" for score in review_score.index]
            )

plt.title("Rating by customers for service", fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(fontsize=12)
st.pyplot(fig)


# Total Sales by Month - Year
st.subheader("Total Sales by Month - Year")

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=sales_long_format_df, x='order_purchase_month', y='total_sales', hue='order_purchase_year', palette='summer_r')

plt.title("Total Sales by Month and Year", fontsize=15)
plt.xlabel("Month")
plt.ylabel("Amount - BRL")
plt.xticks(fontsize=12)
plt.legend(title='Year')
st.pyplot(fig)

# Customer Satisfaction
st.subheader("Customer Satisfaction in 2018")

colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

tab1, tab2 = st.tabs(["Score Distribution", "Positive/Negative"])

with tab1:
    plt.figure(figsize=(16, 8))  
    
    # Filter data for the year 2018
    format = '%Y-%m-%d'
    start_dt = datetime.strptime('2018-01-01', format)
    end_dt = datetime.strptime('2018-12-31', format)
    
    all_df['review_creation_date']=pd.to_datetime(all_df['review_creation_date'], format='mixed')

    data_2018 = all_df[(all_df['review_creation_date'] >= start_dt) & (all_df['review_creation_date'] < end_dt)]

    # finding the last month of available data 2018
    last_month = data_2018['review_creation_date'].max().month

    # list of months
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # availble months from the 2018 data
    available_months = months[0:last_month]

    # Create a bar plot for review scores by month
    sns.countplot(x=data_2018['review_creation_date'].dt.month,
                hue=data_2018['review_score'],
                palette="summer_r")

    plt.title("Customer Satisfaction in 2018", fontsize=15)
    plt.xlabel("Month")
    plt.ylabel("Count of Reviews")
    plt.legend(title="Review Score", loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.xticks(range(0, 8), available_months)

    st.pyplot(plt)  

with tab2:
    plt.figure(figsize=(16, 8))
    
    # Filter data for the year 2018
    format = '%Y-%m-%d'
    start_dt = datetime.strptime('2018-01-01', format)
    end_dt = datetime.strptime('2018-12-31', format)
    
    all_df['review_creation_date']=pd.to_datetime(all_df['review_creation_date'], format='mixed')

    data_2018 = all_df[(all_df['review_creation_date'] >= start_dt) & (all_df['review_creation_date'] < end_dt)]

    # Classify review to 1 and 0 value. If review more than 3 classify to 1, else 0.
    data_2018['classify_score'] = data_2018['review_score'].apply(lambda x: 1 if x > 3 else 0)

    # plotting customer satisaction in 2018 into pie chart
    plt.figure(figsize=(5,5))
    colors = ['darkgreen', 'y','lightgoldenrodyellow','olivedrab', ]
    data_2018['classify_score'].map({0:'Negative',1:'Positive'}).value_counts().plot.pie(autopct='%.2f%%', textprops={'fontsize':12}, startangle=165, colors=colors)
    plt.title('Review Scores - 2018', loc="center", fontsize=15)
    plt.ylabel('')
 
    st.pyplot(plt)

# RFM
st.subheader("RFM Analysis")

tab1, tab2 = st.tabs(["By Class", "By Segmentation"])

with tab1:
    fig, ax = plt.subplots(figsize=(16, 8))

    # Visualize RFM Segments
    sns.scatterplot(x='Recency', y='Frequency', hue='Segment', data=rfm_df)
    
    ax.set_title("RFM Segmentation", loc="center", fontsize=30)
    ax.set_ylabel('Frequency (Number of Purchases)')
    ax.set_xlabel('Recency (Days Since Last Purchase)')
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(16, 8))

    # Calculate average values for each RFM_Level, and return a size of each segment 
    rfm_level_agg = rfm_df.groupby('Label').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(1)

    # remove 1 levels from an index
    rfm_level_agg.columns = rfm_level_agg.columns.droplevel()

    # Rename Columns
    rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']

    # Data
    labels = rfm_level_agg.index
    sizes = rfm_level_agg['Count']
    #colors=['#f0f0f0','#d2d2d2','#b4b4b4','#a5a5a5','#969696','#425a90','#2e4884']
    colors=['#d2d2d2','#b4b4b4','#a5a5a5','#969696','#425a90','#2e4884']

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"))

    # wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.5), startangle=-40)
    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.5), startangle=-40, colors=colors)

    # Inner circle
    centre_circle = plt.Circle((0,0),0.40,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Add labels
    ax.legend(wedges, labels,
        title="Segmentation",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(texts, size=12, weight="bold")

    ax.set_title("RFM Segmentation")

    st.pyplot(fig)

st.caption('Copyright © DRI 2024')

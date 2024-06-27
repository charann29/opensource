import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Used Laptop Price Analysis")
st.sidebar.title("Used Laptop Price Analysis")
st.sidebar.markdown(
    "This is an interactive dashboard to analyze used laptop price data. 💻")

# path to the data file 
data_url = ('df_clean_final.csv')


# only rerun the function if the code changed or will reuse cached data
@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv(data_url).drop(columns='Unnamed: 0')
    return data

data = load_data()

# Show the top 20 lines of the data as a snapshot
st.markdown("### Used Laptop Price Data Snapshot 📸")
st.write(data.head(20))
   

st.markdown("### Sunburst Chart ☀️")
fig = px.sunburst(data, path=['Brand', 'os_type', 'condition'], values='price',
                  color='SSD Capacity', hover_data=['Storage Type'], height=500)
st.plotly_chart(fig)
   
# Create a new price_range column
price_range = []
for p in data['price']:
    if p < 500:
        price_range.append('Under 500')
    elif p > 1000:
        price_range.append('More than 1000')
    else:
        price_range.append('Between 500 - 1000')
data['price_range'] = price_range 
        

st.sidebar.subheader("Show some random laptop")
random_laptop = st.sidebar.radio(
    'Price Range', ('Under 500', 'Between 500 - 1000', 'More than 1000'))
st.sidebar.markdown(data.query('price_range == @random_laptop')[['title']].sample(n=1).iat[0, 0])

st.sidebar.markdown('### Price distribution')
select = st.sidebar.selectbox(
    'Visualization Type', ['Histogram', 'Violin Plot'], key='1')

if not st.sidebar.checkbox('Hide', True):
    st.markdown("### Used laptop price distribution")
    if select == 'Histogram':
        fig = px.histogram(data, x="price")
        st.plotly_chart(fig)
    else:
        fig = px.violin(data, y="price")
        st.plotly_chart(fig)

st.sidebar.markdown('### Laptop counts by storage type')
select_os = st.sidebar.selectbox('Visualization Type', ['Bar chart', 'Pie chat'], key='2')
laptop_count = data['Storage Type'].value_counts()
laptop_count = pd.DataFrame({'Storage Type': laptop_count.index, 'Count': laptop_count.values})
if not st.sidebar.checkbox('Hide', True, key='3'):
    st.markdown('### Laptop counts by storage type')
    if select_os == 'Bar chart':
        fig = px.bar(laptop_count, x='Storage Type', y='Count', color='Count', height=500)
        st.plotly_chart(fig)
    else: 
        fig = px.pie(laptop_count, values='Count', names='Storage Type', height=500)
        st.plotly_chart(fig)
        
st.sidebar.markdown('### Breandown brand by operating system type')
choice= st.sidebar.multiselect('Pick os type(s)', ('windows', 'linux', 'ubuntu', 'mac os'), key='4')
if len(choice) > 0:
    choice_data = data[data['os_type'].isin(choice)]
    fig_choice = px.histogram(choice_data, x='os_type', y='Brand', histfunc='count', color='Brand', facet_col='Brand',
                              labels={'OS Type':'Brand'}, height=600, width=800)
    st.plotly_chart(fig_choice)
           
st.sidebar.markdown('### Scatter plot price vs variable')
select_variable = st.sidebar.selectbox('Select variable', ['RAM Size', 'SSD Capacity', 'Processor Speed'], key='6')
if not st.sidebar.checkbox('Hide', True, key='7'):
    st.markdown('### Scatter plot price vs variable')
    if select_variable == 'RAM Size':
        fig = px.scatter(data, x='RAM Size', y='price', trendline="ols", color="os_type",
                         hover_name="Brand", log_x=True, size_max=60)
        st.plotly_chart(fig)
    elif select_variable == 'SSD Capacity':
        fig = px.scatter(data, x="SSD Capacity", y="price", trendline="ols", color="os_type",
                         hover_name="Brand", log_x=True, size_max=60)
        st.plotly_chart(fig)
    else: 
        fig = px.scatter(data, x='Processor Speed', y='price', trendline="ols", color="os_type",
                         hover_name="Brand", log_x=True, size_max=60)
        st.plotly_chart(fig)

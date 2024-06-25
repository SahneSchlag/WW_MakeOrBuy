import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title('Make-or-Buy Decision Tool - Matthias Krüger')
st.header('Produced Quantities and Batch Sizes')
num_batches = st.number_input('Number of Batches', min_value=1, value=1, step=1)
batch_sizes = [st.number_input(f'Quantity for Batch {i+1}', min_value=1, value=100, step=1, key=f"batch_size_{i}") for i in range(num_batches)]

st.header('Make Costs')
if "make_costs" not in st.session_state:
    st.session_state.make_costs = []

if st.button("Add Make Cost"):
    st.session_state.make_costs.append({"title": "", "type": "Fixed Costs", "value": 0.0, "batches": [], "start_batch": 1, "end_batch": 1})

make_costs_total = np.zeros(len(batch_sizes), dtype=float)
make_constant_costs = np.zeros(len(batch_sizes), dtype=float)

for idx, cost in enumerate(st.session_state.make_costs):
    cols = st.columns(6)
    cost["title"] = cols[0].text_input(f'Title for Make Cost {idx+1}', value=cost.get("title", ""), key=f"make_title_{idx}")
    cost["type"] = cols[1].selectbox(f'Type for Make Cost {idx+1}', ['Fixed Costs', 'Unit Costs'], index=['Fixed Costs', 'Unit Costs'].index(cost["type"]), key=f"make_type_{idx}")
    cost["value"] = cols[2].text_input(f'Price for Make Cost {idx+1}', value=str(cost.get("value", 0.0)), key=f"make_value_{idx}")

    try:
        cost_value = float(cost["value"].replace(",", "."))
    except ValueError:
        cost_value = 0.0

    if cost["type"] == 'Fixed Costs':
        selected_batches = cols[3].multiselect(f'Batch Range for Make Cost {idx+1}', list(range(1, num_batches + 1)), default=[1])
        cost["batches"] = selected_batches
        for batch in selected_batches:
            make_constant_costs[batch - 1] += cost_value
    else:
        cost["start_batch"] = cols[4].number_input(f'From Batch', min_value=1, max_value=num_batches, value=cost.get("start_batch", 1), step=1, key=f"make_start_batch_{idx}")
        cost["end_batch"] = cols[5].number_input(f'To Batch', min_value=1, max_value=num_batches, value=cost.get("end_batch", 1), step=1, key=f"make_end_batch_{idx}")
        for i in range(cost["start_batch"] - 1, cost["end_batch"]):
            make_costs_total[i] += cost_value * batch_sizes[i]

    if cols[5].button(f'Remove Make Cost {idx+1}'):
        del st.session_state.make_costs[idx]
        st.experimental_rerun()

make_data = []
make_cumulative_cost = 0
make_cumulative_pieces = 0
cumulative_costs = [0]
cumulative_pieces = [0]

batch_ranges = [sum(batch_sizes[:i+1]) for i in range(len(batch_sizes))]

for i, batch_size in enumerate(batch_sizes):
    make_variable_cost = make_costs_total[i]
    make_constant_cost = make_constant_costs[i]
    make_total_cost = make_variable_cost + make_constant_cost
    make_cumulative_cost += make_total_cost
    make_cumulative_pieces += batch_size

    row = {
        'Batch': i + 1,
        'Quantity': batch_size,
        'Make Fixed Costs': make_constant_cost,
        'Make Unit Costs': make_variable_cost,
        'Make Total': make_total_cost,
    }

    make_data.append(row)
    cumulative_pieces.append(cumulative_pieces[-1] + batch_size)
    cumulative_costs.append(cumulative_costs[-1] + make_variable_cost + make_constant_cost)

make_df = pd.DataFrame(make_data)

make_totals = make_df[['Make Fixed Costs', 'Make Unit Costs', 'Make Total']].sum()
make_totals_row = pd.DataFrame([{
    'Batch': 'Total',
    'Quantity': make_df['Quantity'].sum(),
    'Make Fixed Costs': make_totals['Make Fixed Costs'],
    'Make Unit Costs': make_totals['Make Unit Costs'],
    'Make Total': make_totals['Make Total']
}])

make_df = pd.concat([make_df, make_totals_row], ignore_index=True)
st.write("Make Costs:")
st.write(make_df)

st.header('Buy Providers and Costs')

if "buy_providers" not in st.session_state:
    st.session_state.buy_providers = []

if st.button("Add Buy Provider"):
    st.session_state.buy_providers.append({"title": "", "costs": []})

for provider_idx, provider in enumerate(st.session_state.buy_providers):
    provider["title"] = st.text_input(f'Title for Buy Provider {provider_idx + 1}', provider.get("title", ""), key=f"provider_title_{provider_idx}")

    if st.button(f"Delete Buy Provider {provider_idx + 1}"):
        del st.session_state.buy_providers[provider_idx]
        st.experimental_rerun()

    if st.button(f"Add Cost for Provider {provider_idx + 1}"):
        provider["costs"].append({"title": "", "type": "Fixed Costs", "value": 0.0, "batches": [], "start_batch": 1, "end_batch": 1})

    buy_costs_total = np.zeros(len(batch_sizes), dtype=float)

    for cost_idx, cost in enumerate(provider["costs"]):
        cols = st.columns(6)
        cost["title"] = cols[0].text_input(f'Title for Cost {cost_idx + 1} of Provider {provider_idx + 1}', value=cost.get("title", ""), key=f"buy_title_{provider_idx}_{cost_idx}")
        cost["type"] = cols[1].selectbox(f'Type for Cost {cost_idx + 1} of Provider {provider_idx + 1}', ['Fixed Costs', 'Unit Costs'], index=['Fixed Costs', 'Unit Costs'].index(cost["type"]), key=f"buy_type_{provider_idx}_{cost_idx}")
        cost["value"] = cols[2].text_input(f'Price for Cost {cost_idx + 1} of Provider {provider_idx + 1}', value=str(cost.get("value", 0.0)), key=f"buy_value_{provider_idx}_{cost_idx}")

        try:
            cost_value = float(cost["value"].replace(",", "."))
        except ValueError:
            cost_value = 0.0

        if cost["type"] == 'Fixed Costs':
            selected_batches = cols[3].multiselect(f'Batch Range for Cost {cost_idx + 1} of Provider {provider_idx + 1}', list(range(1, num_batches + 1)), default=[1])
            cost["batches"] = selected_batches
            for batch in selected_batches:
                buy_costs_total[batch - 1] += cost_value
        else:
            cost["start_batch"] = cols[4].number_input(f'From Batch', min_value=1, max_value=num_batches, value=cost.get("start_batch", 1), step=1, key=f"buy_start_batch_{provider_idx}_{cost_idx}")
            cost["end_batch"] = cols[5].number_input(f'To Batch', min_value=1, max_value=num_batches, value=cost.get("end_batch", 1), step=1, key=f"buy_end_batch_{provider_idx}_{cost_idx}")
            for i in range(cost["start_batch"] - 1, cost["end_batch"]):
                buy_costs_total[i] += cost_value * batch_sizes[i]

        if cols[5].button(f'Remove Cost {cost_idx + 1} of Provider {provider_idx + 1}'):
            del provider["costs"][cost_idx]
            st.experimental_rerun()

    buy_data = []
    buy_cumulative_cost = 0
    buy_cumulative_pieces = [0]
    buy_cumulative_costs = [0]

    for i, batch_size in enumerate(batch_sizes):
        buy_variable_cost = buy_costs_total[i]
        buy_total_cost = buy_variable_cost
        buy_cumulative_cost += buy_total_cost
        buy_cumulative_pieces.append(buy_cumulative_pieces[-1] + batch_size)
        buy_cumulative_costs.append(buy_cumulative_costs[-1] + buy_variable_cost)

        row = {
            'Batch': i + 1,
            'Quantity': batch_size,
            'Buy Fixed Costs': 0.0,
            'Buy Unit Costs': buy_variable_cost,
            'Buy Total': buy_total_cost,
        }

        buy_data.append(row)

    buy_df = pd.DataFrame(buy_data)

    buy_totals = buy_df[['Buy Fixed Costs', 'Buy Unit Costs', 'Buy Total']].sum()
    buy_totals_row = pd.DataFrame([{
        'Batch': 'Total',
        'Quantity': buy_df['Quantity'].sum(),
        'Buy Fixed Costs': buy_totals['Buy Fixed Costs'],
        'Buy Unit Costs': buy_totals['Buy Unit Costs'],
        'Buy Total': buy_totals['Buy Total']
    }])

    buy_df = pd.concat([buy_df, buy_totals_row], ignore_index=True)
    st.write(f"Buy Costs for Provider {provider_idx + 1}:")
    st.write(buy_df)

    plt.plot(buy_cumulative_pieces, buy_cumulative_costs, marker='o', linestyle='-', label=f'Buy Total Costs ({provider["title"]})')

st.header('Cost Comparison Make vs. Buy')

make_cumulative_costs_with_jumps = [0]
cumulative_pieces_with_jumps = [0]

make_cumulative_cost = 0
for i, batch_size in enumerate(batch_sizes):
    make_cumulative_cost += make_constant_costs[i] 
    cumulative_pieces_with_jumps.append(sum(batch_sizes[:i]))
    make_cumulative_costs_with_jumps.append(make_cumulative_cost)
    
    make_cumulative_cost += make_costs_total[i] 
    cumulative_pieces_with_jumps.append(sum(batch_sizes[:i+1]))
    make_cumulative_costs_with_jumps.append(make_cumulative_cost)

plt.plot(cumulative_pieces_with_jumps, make_cumulative_costs_with_jumps, marker='o', linestyle='-', label='Make Total Costs')
plt.xlabel('Quantity')
plt.ylabel('Total Costs (€)')
plt.legend()
plt.title('Cost Comparison Make vs. Buy')
st.pyplot(plt)

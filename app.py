import streamlit as st
import numpy as np
import pandas as pd

# Titel und Beschreibung
st.title('Make-or-Buy Entscheidungstool')
st.write("""
    Dieses Tool hilft Ihnen bei der Entscheidung, ob Sie ein Produkt selbst herstellen (Make) oder einkaufen (Buy) sollten.
    Geben Sie die entsprechenden Kosten und Mengen ein, um die Kostenverläufe anzuzeigen und eine fundierte Entscheidung zu treffen.
""")

# Eingabe der produzierten Stückzahlen und Lose
st.header('Produzierte Stückzahlen und Losgrößen')
num_batches = st.number_input('Anzahl der Lose', min_value=1, value=1, step=1)
batch_sizes = [st.number_input(f'Stückzahl für Los {i+1}', min_value=1, value=100, step=1, key=f"batch_size_{i}") for i in range(num_batches)]

# Eingabe der Make-Kosten
st.header('Make-Kosten')
if "make_costs" not in st.session_state:
    st.session_state.make_costs = []

if st.button("Make-Kosten hinzufügen"):
    st.session_state.make_costs.append({"title": "", "type": "Mengenkonstante Kosten", "value": 0.0, "batch": 1, "start_batch": 1, "end_batch": 1})

make_costs_total = np.zeros(sum(batch_sizes), dtype=float)

for idx, cost in enumerate(st.session_state.make_costs):
    cols = st.columns(6)
    cost["title"] = cols[0].text_input(f'Titel für Make-Kosten {idx+1}', value=cost.get("title", ""), key=f"make_title_{idx}")
    cost["type"] = cols[1].selectbox(f'Typ für Make-Kosten {idx+1}', ['Mengenkonstante Kosten', 'Stückkosten'], index=['Mengenkonstante Kosten', 'Stückkosten'].index(cost["type"]), key=f"make_type_{idx}")
    cost["value"] = cols[2].number_input(f'Preis für Make-Kosten {idx+1}', min_value=0.0, value=cost.get("value", 0.0), step=0.1, key=f"make_value_{idx}")

    if cost["type"] == 'Mengenkonstante Kosten':
        cost["batch"] = cols[3].number_input(f'Los für Make-Kosten {idx+1}', min_value=1, max_value=num_batches, value=cost.get("batch", 1), step=1, key=f"make_batch_{idx}")
        # Berechnung der Mengenkonstanten Kosten nur für den angegebenen Batch
        if cost["batch"] <= num_batches:
            make_costs_total[sum(batch_sizes[:cost["batch"]-1]):sum(batch_sizes[:cost["batch"]])] += cost["value"]
    else:
        cost["start_batch"] = cols[4].number_input(f'Von Los', min_value=1, max_value=num_batches, value=cost.get("start_batch", 1), step=1, key=f"make_start_batch_{idx}")
        cost["end_batch"] = cols[5].number_input(f'Bis Los', min_value=1, max_value=num_batches, value=cost.get("end_batch", 1), step=1, key=f"make_end_batch_{idx}")
        for i in range(cost["start_batch"] - 1, cost["end_batch"]):
            make_costs_total[sum(batch_sizes[:i]):sum(batch_sizes[:i + 1])] += cost["value"]

    if cols[5].button(f'Make-Kosten {idx+1} entfernen'):
        del st.session_state.make_costs[idx]
        st.experimental_rerun()

# Eingabe der Buy-Anbieter und Kosten
st.header('Buy-Anbieter und Kosten')
if "buy_providers" not in st.session_state:
    st.session_state.buy_providers = []

if st.button("Buy-Anbieter hinzufügen"):
    st.session_state.buy_providers.append({"title": "", "costs": []})

buy_costs_total_all_providers = []

for provider_idx, provider in enumerate(st.session_state.buy_providers):
    st.subheader(f"Kosten für Buy-Anbieter {provider_idx + 1}")
    provider["title"] = st.text_input(f'Titel für Buy-Anbieter {provider_idx + 1}', provider.get("title", ""))

    buy_costs_total = np.zeros(sum(batch_sizes), dtype=float)

    for cost_idx, cost in enumerate(provider["costs"]):
        cols = st.columns(6)
        cost["title"] = cols[0].text_input(f'Titel für Kosten {cost_idx + 1}', value=cost.get("title", ""), key=f"buy_title_{provider_idx}_{cost_idx}")
        cost["type"] = cols[1].selectbox(f'Typ für Kosten {cost_idx + 1}', ['Mengenkonstante Kosten', 'Stückkosten'], index=['Mengenkonstante Kosten', 'Stückkosten'].index(cost["type"]), key=f"buy_type_{provider_idx}_{cost_idx}")
        cost["value"] = cols[2].number_input(f'Preis für Kosten {cost_idx + 1}', min_value=0.0, value=cost.get("value", 0.0), step=0.1, key=f"buy_value_{provider_idx}_{cost_idx}")

        if cost["type"] == 'Mengenkonstante Kosten':
            cost["batch"] = cols[3].number_input(f'Los für Kosten {cost_idx + 1}', min_value=1, max_value=num_batches, value=cost.get("batch", 1), step=1, key=f"buy_batch_{provider_idx}_{cost_idx}")
            # Berechnung der Mengenkonstanten Kosten nur für den angegebenen Batch
            if cost["batch"] <= num_batches:
                buy_costs_total[sum(batch_sizes[:cost["batch"]-1]):sum(batch_sizes[:cost["batch"]])] += cost["value"]
        else:  # Stückkosten
            cost["start_batch"] = cols[4].number_input(f'Von Los', min_value=1, max_value=num_batches, value=cost.get("start_batch", 1), step=1, key=f"buy_start_batch_{provider_idx}_{cost_idx}")
            cost["end_batch"] = cols[5].number_input(f'Bis Los', min_value=1, max_value=num_batches, value=cost.get("end_batch", 1), step=1, key=f"buy_end_batch_{provider_idx}_{cost_idx}")
            for i in range(cost["start_batch"] - 1, cost["end_batch"]):
                buy_costs_total[sum(batch_sizes[:i]):sum(batch_sizes[:i + 1])] += cost["value"]

        if cols[5].button(f'Kosten {cost_idx + 1} entfernen'):
            del provider["costs"][cost_idx]
            st.experimental_rerun()

    buy_costs_total_all_providers.append(buy_costs_total)

    buy_data = []
    for i, batch_size in enumerate(batch_sizes):
        batch_end = sum(batch_sizes[:i + 1])
        buy_variable_cost = buy_costs_total[batch_end - batch_size:batch_end].sum()
        buy_total_cost = buy_variable_cost

        row = {
            'Los': i + 1,
            'Stückzahl': batch_size,
            'Buy Mengenkonstante Kosten': 0.0,  # Initialisierung mit 0
            'Buy Stückkosten': buy_variable_cost,
            'Buy Gesamt': buy_total_cost,
        }

        # Setze die Mengenkonstanten Kosten nur für das spezifische Los
        for cost in provider["costs"]:
            if cost["type"] == 'Mengenkonstante Kosten' and cost["batch"] == i + 1:
                row['Buy Mengenkonstante Kosten'] = cost["value"]
                break

        buy_data.append(row)

    buy_df = pd.DataFrame(buy_data)
    st.write(f"Buy-Kosten für Anbieter {provider['title']}:")
    st.write(buy_df)

# Berechnung der Gesamtkosten für Make
quantities = np.arange(1, sum(batch_sizes) + 1)
make_data = []

# Initialize cumulative constant costs array
make_constant_costs = np.zeros(sum(batch_sizes), dtype=float)

# Sum up constant costs for each batch
for cost in st.session_state.make_costs:
    if cost["type"] == 'Mengenkonstante Kosten' and cost["batch"] <= num_batches:
        make_constant_costs[sum(batch_sizes[:cost["batch"]-1]):sum(batch_sizes[:cost["batch"]])] += cost["value"]

# Calculate total costs for each batch
for i, batch_size in enumerate(batch_sizes):
    batch_end = sum(batch_sizes[:i + 1])
    make_variable_cost = make_costs_total[batch_end - batch_size:batch_end].sum()
    make_constant_cost = make_constant_costs[batch_end - batch_size:batch_end].sum()
    make_total_cost = make_variable_cost + make_constant_cost

    row = {
        'Los': i + 1,
        'Stückzahl': batch_size,
        'Make Mengenkonstante Kosten': make_constant_cost / batch_size,  # Divide by batch_size to get per unit cost
        'Make Stückkosten': make_variable_cost / batch_size,  # Divide by batch_size to get per unit cost
        'Make Gesamt': make_total_cost / batch_size,  # Divide by batch_size to get per unit cost
    }

    make_data.append(row)

make_df = pd.DataFrame(make_data)
st.write("Make-Kosten:")
st.write(make_df)

# Plotten der Daten
st.header('Kostenvergleich')
st.subheader('Gesamtkostenverlauf')

quantities = np.arange(1, sum(batch_sizes) + 1)
make_cumulative_costs = np.cumsum(make_costs_total)

data = {
    'Stückzahl': quantities,
    'Make Gesamtkosten': make_cumulative_costs
}

for provider_idx, provider in enumerate(st.session_state.buy_providers):
    buy_cumulative_costs = np.cumsum(buy_costs_total_all_providers[provider_idx])
    data[f'Buy - {provider["title"]} Gesamtkosten'] = buy_cumulative_costs

chart_data = pd.DataFrame(data)

st.line_chart(chart_data.set_index('Stückzahl'))



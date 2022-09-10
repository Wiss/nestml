# Loading data

## examples

```
PATH_TO_LOAD = os.path.join(
                'results',
                f"ed_{network_layout['energy_dependent']}",
                f"2022_09_10_174400_seed_{general['seed']}",
                'data')
s = load_data(PATH_TO_LOAD, 'spikes')
m = load_data(PATH_TO_LOAD, 'multimeter')
w = load_data(PATH_TO_LOAD, 'weights')
```

# Exploring data

## examples

```
print('spikes')
print(spikes)
print(type(spikes))
s_events_dict = {}
for k, v in spikes.items():
print(f'{k} pop')
if spikes[k] is not None:
        s_events = spikes[k].get('events')
        s_events_dict[k] = s_events
        print('spikes senders')
        print(s_events['senders'])
        print('spikes times')
        print(s_events['times'])
print('multimeter')
print(multimeter)
m_events_dict = {}
for k, v in multimeter.items():
print(f'{k} pop')
if multimeter[k] is not None:
        m_events = multimeter[k].get('events')
        m_events_dict[k] = m_events
        print('ATP senders')
        print(m_events['senders'])
        print('ATP vals')
        print(m_events['ATP'])
        print('ATP times')
        print(m_events['times'])
print('weights')
print(weights)
w_events_dict = {}
for k, v in weights.items():
print(f'{k.split("_")[0]} -> {k.split("_")[1]} pop')
if weights[k] is not None:
        w_events = weights[k].get('events')
        w_events_dict[k] = w_events
        print('weight sender')
        print(w_events['senders'])
        print(w_events['targets'])
        print('weights vals')
        print(w_events['weights'])
        print('weights times')
        print(w_events['times'])
```

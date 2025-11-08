const {useState, useEffect, useMemo, useCallback, useRef} = React;

const createId = () => Math.random().toString(36).slice(2, 10);

const defaultSeedTemplate = `deck: 12 carnivorous, 6 sharp
phase: feeding
food: 2
players:
  - hand: 1 sharp, 1 camo
    continent: carn sharp, carn camo
  - hand: 1 sharp, 1 camo
    continent: carn sharp, carn camo
`;

const defaultSettings = () => ({
  name: '',
  maxPlayers: 4,
  timeTurn: 60,
  timeTraitResponse: 30,
  addon_plantarium: false,
  password: ''
});

const createPlayer = (index = 0) => ({
  id: createId(),
  name: `Player ${index + 1}`,
  hand: [],
  continent: ''
});

const createCardToken = (meta) => ({
  uid: createId(),
  seedName: meta && meta.seedName ? meta.seedName : (meta || 'card'),
  name: meta && meta.name ? meta.name : (meta && meta.seedName ? meta.seedName : 'Card')
});

const useSortable = (ref, options) => {
  useEffect(() => {
    if (!ref.current) return;
    const sortable = new Sortable(ref.current, options);
    return () => sortable.destroy();
  }, [ref, options]);
};

const getListRef = (draft, listId) => {
  if (!listId) return null;
  if (listId === 'deck') return draft.deck;
  if (listId === 'deckPlants') return draft.deckPlants;
  const match = listId.match(/^player-(.+?)-hand$/);
  if (match) {
    const player = draft.players.find((p) => p.id === match[1]);
    return player ? player.hand : null;
  }
  return null;
};

const formatSettingValue = (value) => {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') return value;
  return `${value}`;
};

const buildCardString = (cards, lookup) => {
  if (!cards || cards.length === 0) return '';
  const counter = cards.reduce((acc, card) => {
    const key = card.seedName || card.name.toLowerCase();
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  return Object.entries(counter)
    .map(([slug, count]) => {
      const meta = lookup.get(slug) || {seedName: slug};
      const token = meta.seedName || slug;
      return count > 1 ? `${count} ${token}` : `${token}`;
    })
    .join(', ');
};

const resolveFromLibrary = (slug, libraryList) => {
  if (!slug) return null;
  const normalized = slug.trim().toLowerCase();
  return libraryList.find((item) => item.seedName === normalized || item.search.includes(normalized)) || null;
};

const expandCardList = (raw, libraryList) => {
  if (!raw) return [];
  let items = [];
  if (Array.isArray(raw)) {
    items = raw;
  } else if (typeof raw === 'string') {
    items = raw.split(',').map((token) => token.trim()).filter(Boolean);
  } else {
    return [];
  }

  const expanded = [];
  items.forEach((token) => {
    if (!token) return;
    const parts = token.trim().split(/\s+/);
    let count = parseInt(parts[0], 10);
    let slug;
    if (Number.isFinite(count)) {
      slug = parts.slice(1).join(' ');
    } else {
      count = 1;
      slug = token.trim();
    }
    const meta = resolveFromLibrary(slug, libraryList) || {seedName: slug, name: slug};
    for (let i = 0; i < count; i += 1) {
      expanded.push(createCardToken(meta));
    }
  });
  return expanded;
};

const parseSeedString = (seedString, library) => {
  if (!seedString) return null;
  let parsed;
  try {
    parsed = window.jsyaml.load(seedString) || {};
  } catch (err) {
    console.error('Failed to parse seed', err);
    return null;
  }
  const baseSettings = defaultSettings();
  const incomingSettings = parsed.settings || {};
  const customSettings = [];
  Object.entries(incomingSettings).forEach(([key, value]) => {
    if (Object.prototype.hasOwnProperty.call(baseSettings, key)) {
      baseSettings[key] = value;
    } else {
      customSettings.push({id: createId(), key, value: value});
    }
  });

  const players = Array.isArray(parsed.players)
    ? parsed.players.map((player, index) => ({
        id: createId(),
        name: `Player ${index + 1}`,
        hand: expandCardList(player && player.hand, library.cards),
        continent: player && player.continent ? player.continent : ''
      }))
    : [createPlayer(0)];

  return {
    phase: parsed.phase ? String(parsed.phase).toLowerCase() : '',
    food: typeof parsed.food === 'number' ? parsed.food : '',
    deck: expandCardList(parsed.deck, library.cards),
    deckPlants: expandCardList(parsed.deckPlants, library.plants),
    players: players.length ? players : [createPlayer(0)],
    settings: baseSettings,
    customSettings
  };
};

const CardToken = ({card, onRemove}) => (
  <div className="card-token" data-uid={card.uid} data-slug={card.seedName}>
    <span>{card.name}</span>
    {onRemove ? (
      <button
        className="secondary"
        style={{marginLeft: '8px', padding: '4px 8px'}}
        onClick={(event) => {
          event.stopPropagation();
          event.preventDefault();
          onRemove();
        }}
      >
        ×
      </button>
    ) : null}
  </div>
);

const CardDropZone = ({title, description, listId, items, onDrop, onRemoveCard, placeholder, groupName = 'cards'}) => {
  const containerRef = useRef(null);
  useSortable(containerRef, useMemo(() => ({
    group: {name: groupName, pull: true, put: true},
    animation: 150,
    sort: true,
    draggable: '.card-token',
    onEnd: onDrop
  }), [onDrop, groupName]));

  return (
    <div className="drop-zone">
      <div className="section-header" style={{marginBottom: '12px'}}>
        <div>
          <h3 style={{margin: 0}}>{title}</h3>
          {description ? <p className="section-description">{description}</p> : null}
        </div>
      </div>
      <div className="card-list" ref={containerRef} data-list-id={listId}>
        {items && items.length > 0
          ? items.map((card) => (
              <CardToken key={card.uid} card={card} onRemove={onRemoveCard ? () => onRemoveCard(listId, card.uid) : null} />
            ))
          : null}
      </div>
      {(!items || items.length === 0) ? <p style={{color: '#829ab1', marginTop: '12px'}}>{placeholder}</p> : null}
    </div>
  );
};

const CardLibrary = ({title, cards, groupName = 'cards'}) => {
  const [query, setQuery] = useState('');
  const containerRef = useRef(null);
  useSortable(containerRef, useMemo(() => ({
    group: {name: groupName, pull: 'clone', put: false},
    animation: 150,
    sort: false,
    draggable: '.card-token'
  }), [groupName]));

  const filteredCards = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return cards;
    return cards.filter((card) => card.name.toLowerCase().includes(normalized) || card.seedName.includes(normalized));
  }, [query, cards]);

  return (
    <section>
      <div className="section-header">
        <div>
          <h2>{title}</h2>
          <p className="section-description">Drag cards into the deck or player hands.</p>
        </div>
        <span className="badge">{filteredCards.length} cards</span>
      </div>
      <div className="library-search">
        <input
          type="text"
          placeholder="Search cards"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <div className="card-list" ref={containerRef} data-list-id={groupName === 'cards' ? 'library' : 'plant-library'}>
        {filteredCards.map((card) => (
          <div key={card.id} className="card-token" data-slug={card.seedName}>
            {card.name}
          </div>
        ))}
      </div>
    </section>
  );
};

const CustomSettingsEditor = ({settings, onChange}) => (
  <section>
    <div className="section-header">
      <div>
        <h2>Advanced room settings</h2>
        <p className="section-description">Add custom key-value pairs for rare options. Values are exported exactly as entered.</p>
      </div>
      <button className="secondary" onClick={() => onChange('add')}>
        + Add custom setting
      </button>
    </div>
    {settings.length === 0 ? (
      <p style={{color: '#829ab1'}}>No custom settings. Click “Add custom setting” to append one.</p>
    ) : (
      settings.map((entry) => (
        <div key={entry.id} className="grid-two-columns" style={{marginBottom: '12px'}}>
          <div>
            <label>Key</label>
            <input
              type="text"
              value={entry.key}
              onChange={(event) => onChange('update', entry.id, {key: event.target.value})}
            />
          </div>
          <div>
            <label>Value</label>
            <input
              type="text"
              value={entry.value}
              onChange={(event) => onChange('update', entry.id, {value: event.target.value})}
            />
          </div>
          <div style={{display: 'flex', alignItems: 'flex-end'}}>
            <button className="danger" onClick={() => onChange('remove', entry.id)}>Remove</button>
          </div>
        </div>
      ))
    )}
  </section>
);

const buildSettingsEntries = (settings, customSettings) => {
  const entries = Object.entries(settings)
    .filter(([key, value]) => {
      if (typeof value === 'boolean') return value;
      if (value === '' || value === null || value === undefined) return false;
      return true;
    })
    .map(([key, value]) => [key, value]);
  customSettings.forEach((entry) => {
    if (entry.key && entry.key.trim()) {
      entries.push([entry.key.trim(), entry.value]);
    }
  });
  return entries;
};

const buildSeedString = (config, cardLookup, plantLookup) => {
  const lines = [];
  const deckString = buildCardString(config.deck, cardLookup);
  const plantDeckString = buildCardString(config.deckPlants, plantLookup);
  if (deckString) lines.push(`deck: ${deckString}`);
  if (plantDeckString) lines.push(`deckPlants: ${plantDeckString}`);
  if (config.phase) lines.push(`phase: ${config.phase}`);
  if (config.food !== '' && config.food !== null && config.food !== undefined) lines.push(`food: ${config.food}`);
  lines.push('players:');
  config.players.forEach((player) => {
    const handString = buildCardString(player.hand, cardLookup);
    const continent = player.continent ? player.continent.trim() : '';
    if (handString && continent) {
      lines.push(`  - hand: ${handString}`);
      lines.push(`    continent: ${continent}`);
    } else if (handString) {
      lines.push(`  - hand: ${handString}`);
    } else if (continent) {
      lines.push(`  - continent: ${continent}`);
    } else {
      lines.push('  - {}');
    }
  });

  const settingsEntries = buildSettingsEntries(config.settings, config.customSettings);
  if (settingsEntries.length) {
    lines.push('settings:');
    settingsEntries.forEach(([key, value]) => {
      lines.push(`  ${key}: ${formatSettingValue(value)}`);
    });
  }
  return lines.join('\n');
};

const RoomSettings = ({config, onUpdate, phases}) => (
  <section>
    <div className="section-header">
      <div>
        <h2>Room configuration</h2>
        <p className="section-description">Adjust game settings that are typically edited via the in-room admin panel.</p>
      </div>
    </div>
    <div className="settings-grid">
      <div>
        <label>Phase</label>
        <select value={config.phase} onChange={(event) => onUpdate('phase', event.target.value)}>
          <option value="">Default</option>
          {phases.map((phase) => (
            <option key={phase} value={phase}>{phase}</option>
          ))}
        </select>
      </div>
      <div>
        <label>Food pool</label>
        <input
          type="number"
          value={config.food}
          onChange={(event) => onUpdate('food', event.target.value === '' ? '' : Number(event.target.value))}
        />
      </div>
      <div>
        <label>Room name</label>
        <input
          type="text"
          value={config.settings.name}
          onChange={(event) => onUpdate('settings', {name: event.target.value})}
        />
      </div>
      <div>
        <label>Max players</label>
        <input
          type="number"
          value={config.settings.maxPlayers}
          onChange={(event) => onUpdate('settings', {maxPlayers: event.target.value === '' ? '' : Number(event.target.value)})}
        />
      </div>
      <div>
        <label>Turn time (seconds)</label>
        <input
          type="number"
          value={config.settings.timeTurn}
          onChange={(event) => onUpdate('settings', {timeTurn: event.target.value === '' ? '' : Number(event.target.value)})}
        />
      </div>
      <div>
        <label>Trait response time (seconds)</label>
        <input
          type="number"
          value={config.settings.timeTraitResponse}
          onChange={(event) => onUpdate('settings', {timeTraitResponse: event.target.value === '' ? '' : Number(event.target.value)})}
        />
      </div>
      <div>
        <label>Password</label>
        <input
          type="text"
          value={config.settings.password}
          onChange={(event) => onUpdate('settings', {password: event.target.value})}
        />
      </div>
      <div className="inline" style={{marginTop: '28px'}}>
        <input
          id="addon-plantarium"
          type="checkbox"
          checked={!!config.settings.addon_plantarium}
          onChange={(event) => onUpdate('settings', {addon_plantarium: event.target.checked})}
        />
        <label htmlFor="addon-plantarium">Enable Plantarium addon</label>
      </div>
    </div>
  </section>
);

const PlayersEditor = ({players, onAdd, onRemove, onUpdateName, onUpdateContinent, onRemoveCard, onDrop}) => (
  <section>
    <div className="section-header">
      <div>
        <h2>Players</h2>
        <p className="section-description">Drop cards into each player’s hand and describe their continent setup.</p>
      </div>
      <button className="secondary" onClick={onAdd}>+ Add player</button>
    </div>
    {players.map((player, index) => (
      <div key={player.id} className="player-card">
        <div className="player-header">
          <div>
            <label>Display label</label>
            <input
              type="text"
              value={player.name}
              onChange={(event) => onUpdateName(player.id, event.target.value)}
            />
          </div>
          <button className="danger" onClick={() => onRemove(player.id)} disabled={players.length <= 1}>
            Remove player
          </button>
        </div>
        <CardDropZone
          title={`Starting hand`}
          description={`Player ${index + 1}`}
          listId={`player-${player.id}-hand`}
          items={player.hand}
          onDrop={onDrop}
          onRemoveCard={onRemoveCard}
          placeholder="Drag cards here to give them to the player."
        />
        <div style={{marginTop: '16px'}}>
          <label>Continent configuration (traits, animals, links)</label>
          <textarea
            value={player.continent}
            placeholder="$A TraitCarnivorous, $B TraitSymbiosis$A"
            onChange={(event) => onUpdateContinent(player.id, event.target.value)}
          />
        </div>
      </div>
    ))}
  </section>
);

function App() {
  const [library, setLibrary] = useState({cards: [], traits: [], plants: [], phases: []});
  const [config, setConfig] = useState({
    phase: '',
    food: '',
    deck: [],
    deckPlants: [],
    players: [createPlayer(0)],
    settings: defaultSettings(),
    customSettings: []
  });
  const [exportedSeed, setExportedSeed] = useState('');
  const [importText, setImportText] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const initializedRef = useRef(false);

  useEffect(() => {
    fetch('/api/library')
      .then((response) => response.json())
      .then((data) => {
        setLibrary(data);
      })
      .catch((error) => {
        console.error('Failed to load library', error);
      });
  }, []);

  useEffect(() => {
    if (!initializedRef.current && library.cards.length) {
      initializedRef.current = true;
      const parsed = parseSeedString(defaultSeedTemplate, library);
      if (parsed) {
        setConfig((current) => ({
          ...current,
          ...parsed,
          players: parsed.players,
          settings: {...defaultSettings(), ...parsed.settings},
          customSettings: parsed.customSettings || []
        }));
        setExportedSeed(defaultSeedTemplate.trim());
        setImportText(defaultSeedTemplate.trim());
      }
    }
  }, [library]);

  const cardLookup = useMemo(() => {
    const map = new Map();
    library.cards.forEach((card) => map.set(card.seedName, card));
    return map;
  }, [library.cards]);

  const plantLookup = useMemo(() => {
    const map = new Map();
    library.plants.forEach((plant) => map.set(plant.seedName, plant));
    return map;
  }, [library.plants]);

  const mutateConfig = useCallback((mutator) => {
    setConfig((prev) => {
      const draft = {
        ...prev,
        deck: [...prev.deck],
        deckPlants: [...prev.deckPlants],
        players: prev.players.map((player) => ({
          ...player,
          hand: [...player.hand]
        })),
        settings: {...prev.settings},
        customSettings: prev.customSettings.map((entry) => ({...entry}))
      };
      mutator(draft);
      return draft;
    });
  }, []);

  const handleDrop = useCallback((evt) => {
    const toListId = evt.to.dataset.listId;
    const fromListId = evt.from.dataset.listId;
    if (!toListId) return;

    const slug = evt.item.dataset.slug;
    const uid = evt.item.dataset.uid;
    const newIndex = evt.newIndex;

    const pickLookup = (listId) => (listId === 'deckPlants' || fromListId === 'deckPlants' || listId === 'plant-library') ? plantLookup : cardLookup;

    if (fromListId === 'library' || fromListId === 'plant-library') {
      const lookup = pickLookup(toListId);
      const meta = lookup.get(slug) || {seedName: slug, name: slug};
      mutateConfig((draft) => {
        const list = getListRef(draft, toListId);
        if (!list) return;
        list.splice(newIndex, 0, createCardToken(meta));
      });
      if (evt.item && evt.item.parentNode) {
        evt.item.parentNode.removeChild(evt.item);
      }
      return;
    }

    mutateConfig((draft) => {
      const source = getListRef(draft, fromListId);
      const target = getListRef(draft, toListId);
      if (!source || !target) return;
      const index = source.findIndex((card) => card.uid === uid);
      if (index === -1) return;
      const [card] = source.splice(index, 1);
      target.splice(newIndex, 0, card);
    });
  }, [cardLookup, plantLookup, mutateConfig]);

  const handleRemoveCard = useCallback((listId, uid) => {
    mutateConfig((draft) => {
      const list = getListRef(draft, listId);
      if (!list) return;
      const index = list.findIndex((card) => card.uid === uid);
      if (index !== -1) list.splice(index, 1);
    });
  }, [mutateConfig]);

  const handleAddPlayer = useCallback(() => {
    mutateConfig((draft) => {
      draft.players.push(createPlayer(draft.players.length));
    });
  }, [mutateConfig]);

  const handleRemovePlayer = useCallback((playerId) => {
    mutateConfig((draft) => {
      if (draft.players.length <= 1) return;
      draft.players = draft.players.filter((player) => player.id !== playerId);
    });
  }, [mutateConfig]);

  const handleUpdatePlayerName = useCallback((playerId, value) => {
    mutateConfig((draft) => {
      const player = draft.players.find((p) => p.id === playerId);
      if (player) player.name = value;
    });
  }, [mutateConfig]);

  const handleUpdatePlayerContinent = useCallback((playerId, value) => {
    mutateConfig((draft) => {
      const player = draft.players.find((p) => p.id === playerId);
      if (player) player.continent = value;
    });
  }, [mutateConfig]);

  const handleRoomUpdate = useCallback((field, value) => {
    mutateConfig((draft) => {
      if (field === 'phase') {
        draft.phase = value;
      } else if (field === 'food') {
        draft.food = value;
      } else if (field === 'settings') {
        draft.settings = {...draft.settings, ...value};
      }
    });
  }, [mutateConfig]);

  const handleCustomSettingsChange = useCallback((action, id, payload) => {
    mutateConfig((draft) => {
      if (action === 'add') {
        draft.customSettings.push({id: createId(), key: '', value: ''});
      }
      if (action === 'remove') {
        draft.customSettings = draft.customSettings.filter((entry) => entry.id !== id);
      }
      if (action === 'update') {
        const entry = draft.customSettings.find((item) => item.id === id);
        if (entry) Object.assign(entry, payload);
      }
    });
  }, [mutateConfig]);

  const handleExport = useCallback(() => {
    const seedString = buildSeedString(config, cardLookup, plantLookup);
    setExportedSeed(seedString);
    setStatusMessage('Seed generated.');
    if (navigator.clipboard && seedString) {
      navigator.clipboard.writeText(seedString).then(() => {
        setStatusMessage('Seed copied to clipboard.');
      }).catch(() => {
        setStatusMessage('Seed ready. Copy it manually if needed.');
      });
    }
  }, [config, cardLookup, plantLookup]);

  const handleImport = useCallback(() => {
    if (!importText.trim()) return;
    const parsed = parseSeedString(importText, library);
    if (parsed) {
      setConfig((current) => ({
        ...current,
        ...parsed,
        players: parsed.players,
        settings: {...defaultSettings(), ...parsed.settings},
        customSettings: parsed.customSettings || []
      }));
      setStatusMessage('Seed imported successfully.');
    } else {
      setStatusMessage('Failed to parse provided seed.');
    }
  }, [importText, library]);

  return (
    <div className="app-container">
      <section>
        <div className="section-header">
          <div>
            <h1>Evolution Admin Seed Editor</h1>
            <p className="section-description">
              Build complex room setups visually and export them into the admin command text field. Drag cards into the deck or directly into player hands, tweak room options, and export a ready-to-use YAML seed.
            </p>
          </div>
        </div>
        <div className="grid-two-columns">
          <div>
            <label>Import existing seed</label>
            <textarea
              value={importText}
              onChange={(event) => setImportText(event.target.value)}
              placeholder="Paste seed YAML here and click Import"
              style={{minHeight: '160px'}}
            />
            <div style={{marginTop: '8px', display: 'flex', gap: '12px'}}>
              <button className="secondary" onClick={() => setImportText(defaultSeedTemplate.trim())}>Load default example</button>
              <button className="primary" onClick={handleImport}>Import</button>
            </div>
          </div>
          <div className="export-area">
            <label>Exported seed</label>
            <textarea value={exportedSeed} readOnly placeholder="Click Export to generate seed" />
            <div style={{marginTop: '8px'}}>
              <button className="primary" onClick={handleExport}>Export seed</button>
            </div>
            {statusMessage ? <p style={{color: '#2680c2', marginTop: '8px'}}>{statusMessage}</p> : null}
          </div>
        </div>
      </section>

      <RoomSettings config={config} onUpdate={handleRoomUpdate} phases={library.phases || []} />

      <section>
        <div className="section-header">
          <div>
            <h2>Deck configuration</h2>
            <p className="section-description">Build the main deck that will be used to distribute cards.</p>
          </div>
        </div>
        <CardDropZone
          title="Main deck"
          description="Cards will be exported as a YAML deck entry"
          listId="deck"
          items={config.deck}
          onDrop={handleDrop}
          onRemoveCard={handleRemoveCard}
          placeholder="Drag cards from the library to add them to the deck."
        />
        {config.settings.addon_plantarium ? (
          <div style={{marginTop: '16px'}}>
            <CardDropZone
              title="Plant deck"
              description="Only exported when Plantarium addon is enabled"
              listId="deckPlants"
              items={config.deckPlants}
              onDrop={handleDrop}
              onRemoveCard={handleRemoveCard}
              placeholder="Drag plant cards here."
              groupName="plants"
            />
          </div>
        ) : null}
      </section>

      <PlayersEditor
        players={config.players}
        onAdd={handleAddPlayer}
        onRemove={handleRemovePlayer}
        onUpdateName={handleUpdatePlayerName}
        onUpdateContinent={handleUpdatePlayerContinent}
        onRemoveCard={handleRemoveCard}
        onDrop={handleDrop}
      />

      <CustomSettingsEditor settings={config.customSettings} onChange={handleCustomSettingsChange} />

      <CardLibrary title="Card library" cards={library.cards} groupName="cards" />
      {config.settings.addon_plantarium ? (
        <CardLibrary title="Plant library" cards={library.plants} groupName="plants" />
      ) : null}
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));

import React, {useEffect, useMemo, useState} from 'react';
import {makeStyles} from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';
import Typography from '@material-ui/core/Typography';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Divider from '@material-ui/core/Divider';

const useStyles = makeStyles((theme) => ({
  container: {
    padding: theme.spacing(3)
  },
  paper: {
    padding: theme.spacing(2),
    height: '100%'
  },
  statsGrid: {
    marginBottom: theme.spacing(3)
  },
  statCard: {
    padding: theme.spacing(2),
    height: '100%'
  },
  tableContainer: {
    marginTop: theme.spacing(2)
  },
  tableCellDense: {
    paddingTop: theme.spacing(0.75),
    paddingBottom: theme.spacing(0.75)
  },
  error: {
    color: theme.palette.error.main
  }
}));

const formatNumber = (value) => {
  if (value == null) return '—';
  const numberValue = Number(value);
  if (Number.isNaN(numberValue)) return value;
  return numberValue.toLocaleString();
};

const formatDate = (value) => {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
};

const StatsCard = ({title, value, description}) => {
  const classes = useStyles();
  return (
    <Paper className={classes.statCard} elevation={2}>
      <Typography variant='subtitle2' color='textSecondary'>
        {title}
      </Typography>
      <Typography variant='h4'>
        {formatNumber(value)}
      </Typography>
      {description && (
        <Typography variant='body2' color='textSecondary'>
          {description}
        </Typography>
      )}
    </Paper>
  );
};

const SummaryTable = ({title, rows, columns, emptyLabel = 'No data'}) => {
  const classes = useStyles();
  return (
    <Paper className={classes.paper} elevation={2}>
      <Typography variant='h6'>{title}</Typography>
      <Divider style={{margin: '16px 0'}} />
      <div className={classes.tableContainer}>
        <Table size='small'>
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell key={column.key}>{column.label}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={columns.length}>{emptyLabel}</TableCell>
              </TableRow>
            )}
            {rows.map((row, index) => (
              <TableRow key={row.key || index}>
                {columns.map((column) => (
                  <TableCell
                    key={column.key}
                    className={classes.tableCellDense}
                  >
                    {column.render ? column.render(row[column.key], row) : row[column.key]}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Paper>
  );
};

const extractRecentContext = (context) => {
  if (!context) return '—';
  try {
    if (typeof context === 'string') return context;
    return JSON.stringify(context);
  } catch (error) {
    return '—';
  }
};

const AnalyticsDashboard = () => {
  const classes = useStyles();
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch('/api/analytics/summary')
      .then((response) => {
        if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
        return response.json();
      })
      .then((data) => {
        if (!cancelled) {
          setSummary(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const usersOverTime = useMemo(() => {
    if (!summary || !summary.usersOverTime) return [];
    return summary.usersOverTime.slice(-14).map((entry) => ({
      ...entry,
      count: formatNumber(entry.count)
    }));
  }, [summary]);

  const gamesOverTime = useMemo(() => {
    if (!summary || !summary.gamesOverTime) return [];
    return summary.gamesOverTime.slice(-14).map((entry) => ({
      ...entry,
      count: formatNumber(entry.count)
    }));
  }, [summary]);

  const actionsByType = useMemo(() => {
    if (!summary || !summary.actionsByType) return [];
    return summary.actionsByType.slice(0, 15).map((entry) => ({
      ...entry,
      count: formatNumber(entry.count)
    }));
  }, [summary]);

  const recentEvents = useMemo(() => {
    if (!summary || !summary.recentEvents) return [];
    return summary.recentEvents.map((event, index) => ({
      key: `${event.eventType}-${event.createdAt}-${index}`,
      ...event
    }));
  }, [summary]);

  if (loading) {
    return (
      <Grid className={classes.container} container>
        <Typography variant='h6'>Loading analytics…</Typography>
      </Grid>
    );
  }

  if (error) {
    return (
      <Grid className={classes.container} container>
        <Typography className={classes.error} variant='h6'>
          Failed to load analytics: {error.message}
        </Typography>
      </Grid>
    );
  }

  if (!summary || summary.enabled === false) {
    return (
      <Grid className={classes.container} container>
        <Typography variant='h6'>Analytics service is disabled.</Typography>
      </Grid>
    );
  }

  return (
    <Grid className={classes.container} container spacing={3}>
      <Grid className={classes.statsGrid} container item spacing={3}>
        <Grid item xs={12} sm={4} md={3}>
          <StatsCard
            title='Tracked Events'
            value={summary?.totals?.events}
            description='Total events stored in analytics database'
          />
        </Grid>
        <Grid item xs={12} sm={4} md={3}>
          <StatsCard
            title='Active Users (7 days)'
            value={summary?.totals?.activeUsers7d}
            description='Unique users that interacted within the past week'
          />
        </Grid>
        <Grid item xs={12} sm={4} md={3}>
          <StatsCard
            title='Games In Progress'
            value={summary?.totals?.gamesInProgress}
            description='Games that started but have not finished yet'
          />
        </Grid>
      </Grid>

      <Grid item xs={12} md={6}>
        <SummaryTable
          title='Daily Active Users'
          rows={usersOverTime}
          columns={[{key: 'day', label: 'Day'}, {key: 'count', label: 'Users'}]}
          emptyLabel='No user activity recorded yet.'
        />
      </Grid>

      <Grid item xs={12} md={6}>
        <SummaryTable
          title='Games Started per Day'
          rows={gamesOverTime}
          columns={[{key: 'day', label: 'Day'}, {key: 'count', label: 'Games'}]}
          emptyLabel='No games recorded in the selected period.'
        />
      </Grid>

      <Grid item xs={12} md={6}>
        <SummaryTable
          title='Event Volume by Type'
          rows={actionsByType}
          columns={[{key: 'event_type', label: 'Event Type'}, {key: 'count', label: 'Count'}]}
          emptyLabel='No events logged yet.'
        />
      </Grid>

      <Grid item xs={12} md={6}>
        <SummaryTable
          title='Recent Activity'
          rows={recentEvents}
          columns={[
            {key: 'eventType', label: 'Event'},
            {key: 'login', label: 'User'},
            {key: 'vkLogin', label: 'VK'},
            {key: 'createdAt', label: 'Timestamp', render: (value) => formatDate(value)},
            {key: 'context', label: 'Details', render: (value) => extractRecentContext(value)}
          ]}
          emptyLabel='No recent events logged.'
        />
      </Grid>
    </Grid>
  );
};

export default AnalyticsDashboard;

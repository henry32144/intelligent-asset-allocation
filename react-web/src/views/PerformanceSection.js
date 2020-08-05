import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import { BASEURL } from '../Constants';
import {Doughnut} from 'react-chartjs-2';
import {Line} from 'react-chartjs-2';

const useStyles = makeStyles((theme) => ({
  sectionRoot: {
    margin: theme.spacing(2, 0, 2),
  },
  sectionTitle: {
    margin: theme.spacing(0, 0, 2),
  },
  chartTitle: {
    margin: theme.spacing(2, 0, 2),
  },
}));

export default function PerformanceSection(props) {
  const classes = useStyles();


  return (
    <div className={classes.sectionRoot}>
      <Typography className={classes.sectionTitle} variant="h5">
        Portfolio Weights
      </Typography>
      <Doughnut
        data={props.portfolioWeights}
      />
      <Typography className={classes.chartTitle} variant="h5">
        Performance
      </Typography>
      <Line
        data={props.portfolioPerformances}
      />
      <Typography className={classes.chartTitle} variant="h5">
        History Weights
      </Typography>
      <Line
        data={props.historyWeights}
      />
    </div>
  );
}
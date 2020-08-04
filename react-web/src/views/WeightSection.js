import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import { BASEURL } from '../Constants';
import {Pie} from 'react-chartjs-2';
const useStyles = makeStyles((theme) => ({
  sectionRoot: {
    margin: theme.spacing(2, 0, 2),
  },
  sectionTitle: {
    margin: theme.spacing(0, 0, 2),
  }
}));

export default function PerformanceSection(props) {
  const classes = useStyles();
  

  const data = {
    labels: [
      'GOOG',
      'APPL',
      'AMZN'
    ],
    datasets: [{
      data: [50, 20, 30],
      backgroundColor: [
      '#FF6384',
      '#36A2EB',
      '#FFCE56'
      ],
      hoverBackgroundColor: [
      '#FF6384',
      '#36A2EB',
      '#FFCE56'
      ]
    }]
  };
  return (
    <div className={classes.sectionRoot}>
      <Typography className={classes.sectionTitle} variant="h5">
        Portfolio Weight
      </Typography>
      <Pie data={data} />
    </div>
  );
}
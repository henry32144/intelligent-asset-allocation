import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Typography from '@material-ui/core/Typography';
import { BASEURL } from '../Constants';
import {Line} from 'react-chartjs-2';

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
    labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July'],
    datasets: [
      {
        label: 'GOOG',
        fill: false,
        lineTension: 0.1,
        backgroundColor: '#FF6384',
        borderColor: '#FF6384',
        borderCapStyle: 'butt',
        borderDash: [],
        borderDashOffset: 0.0,
        borderJoinStyle: 'miter',
        pointBorderColor: '#FF6384',
        pointBackgroundColor: '#fff',
        pointBorderWidth: 1,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#FF6384',
        pointHoverBorderColor: '#FF6384',
        pointHoverBorderWidth: 2,
        pointRadius: 1,
        pointHitRadius: 10,
        data: [40, 59, 80, 61, 70, 90, 100]
      },
      {
        label: 'APPL',
        fill: false,
        lineTension: 0.1,
        backgroundColor: '#36A2EB',
        borderColor: '#36A2EB',
        borderCapStyle: 'butt',
        borderDash: [],
        borderDashOffset: 0.0,
        borderJoinStyle: 'miter',
        pointBorderColor: '#36A2EB',
        pointBackgroundColor: '#fff',
        pointBorderWidth: 1,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#36A2EB',
        pointHoverBorderColor: '#36A2EB',
        pointHoverBorderWidth: 2,
        pointRadius: 1,
        pointHitRadius: 10,
        data: [50, 29, 30, 58, 65, 75, 67]
      },
      {
        label: 'AMZN',
        fill: false,
        lineTension: 0.1,
        backgroundColor: '#FFCE56',
        borderColor: '#FFCE56',
        borderCapStyle: 'butt',
        borderDash: [],
        borderDashOffset: 0.0,
        borderJoinStyle: 'miter',
        pointBorderColor: '#FFCE56',
        pointBackgroundColor: '#fff',
        pointBorderWidth: 1,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: '#FFCE56',
        pointHoverBorderColor: '#FFCE56',
        pointHoverBorderWidth: 2,
        pointRadius: 1,
        pointHitRadius: 10,
        data: [66, 64, 70, 75, 63, 80, 79]
      },
      {
        label: 'ADWE',
        fill: false,
        lineTension: 0.1,
        data: [66, 64, 70, 75, 63, 80, 79]
      }
    ]
  };
  return (
    <div className={classes.sectionRoot}>
      <Typography className={classes.sectionTitle} variant="h5">
        Performance
      </Typography>
      <Line data={data} />
    </div>
  );
}
import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import { Line } from 'react-chartjs-2';
import Grid from '@material-ui/core/Grid';
const useStyles = makeStyles((theme) => ({
  sectionRoot: {
    margin: theme.spacing(2, 0, 2),
  },
  sectionTitle: {
    margin: theme.spacing(0, 0, 2),
  },
  incomeTitle: {
    margin: theme.spacing(2, 0, 2),
  },
  chartTitle: {
    margin: theme.spacing(2, 0, 2),
  },
  earnMoneyText: {
    color: "#00873c"
  },
  loseMoneyText: {
    color: "#f0162f"
  },
}));

export default function PerformanceSection(props) {
  const classes = useStyles();

  const getCurrentPerformanceText = () => {
    var performanceText = "";
    var classType = undefined;
    if (props.currentPerformance >= 1.) {
      classType = classes.earnMoneyText;
      performanceText = " " + parseFloat(props.currentPerformance).toFixed(2);
    } else {
      classType = classes.loseMoneyText;
      performanceText = " " + parseFloat(props.currentPerformance).toFixed(2);
    }

    return <Typography variant="h4" display="inline" className={classType} >
      {performanceText}
    </Typography>
  }

  return (
    <div className={classes.sectionRoot}>
      {/* <Typography className={classes.sectionTitle} variant="h5">
        Portfolio Weights
      </Typography>
      <Doughnut
        data={props.portfolioWeights}
      /> */}
      <Grid container className={classes.sectionTitle}>
        <Grid item xs={12} sm={6}>
          <Typography variant="h5" display="inline">
            Estimated value:
          </Typography>
          {getCurrentPerformanceText()}
        </Grid>
      </Grid>
      <Typography className={classes.chartTitle} variant="h6">
        BackTest Performance
      </Typography>
      <Line
        data={props.portfolioPerformances}
      />
      <Typography className={classes.chartTitle} variant="h6">
        History Weights
      </Typography>
      <Line
        data={props.historyWeights}
      />
    </div>
  );
}
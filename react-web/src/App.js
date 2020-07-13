import React from 'react';
import './App.css';
import { makeStyles } from '@material-ui/core/styles';
import TopNavBar from './components/TopNavBar'
import SubmitSelectionButton from './components/SubmitSelectionButton';
import Grid from '@material-ui/core/Grid';
import InvestmentStrategyRadios from './components/InvestmentStrategyRadios'
import Typography from '@material-ui/core/Typography';
import StockSelectSection from './views/StockSelectSection'

const useStyles = makeStyles((theme) => ({
  title: {
    textAlign: 'initial',
    margin: theme.spacing(4, 0, 2),
  },
}));

function App() {
  const classes = useStyles();

  
  return (
    <div className="App">
      <div className={classes.root}>
        <TopNavBar></TopNavBar>
        <Grid container direction="column" justify="center" alignItems="center">
          <Grid className={classes.title} item xs={12}>
            <Typography variant="h6">
              Choose your strategy
            </Typography>
          </Grid>
          <Grid className={classes.gridItem} item xs={12}>
            <InvestmentStrategyRadios/>
          </Grid>
          <Grid className={classes.title} item xs={12}>
            <Typography variant="h6">
              Select Stocks
            </Typography>
          </Grid>
          <Grid className={classes.gridItem} item xs={12} >
            <StockSelectSection></StockSelectSection>
          </Grid>
          <Grid className={classes.gridItem} item xs={12} >
            <SubmitSelectionButton/>
          </Grid>
        </Grid>
      </div>
    </div>
  );
}

export default App;

import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';

import Typography from '@material-ui/core/Typography';
import TopNavBar from '../components/TopNavBar'
import InvestmentStrategyRadios from '../components/InvestmentStrategyRadios'
import SubmitSelectionButton from '../components/SubmitSelectionButton';
import StockSelectSection from '../views/StockSelectSection'

const useStyles = makeStyles((theme) => ({
  title: {
    textAlign: 'initial',
    margin: theme.spacing(4, 0, 2),
  },
}));

function PortfolioPage() {
  const classes = useStyles();
  const [selectedStocks, setSelectedData] = React.useState([
    { companyName: "Alphabet Inc.", companyCode: "NASDAQ：GOOG" },
    { companyName: "Apple Inc.", companyCode: "NASDAQ: AAPL" },
    { companyName: "Amazon.com, Inc.", companyCode: "NASDAQ: AMZN" }
  ]);


  return (
    <div className={classes.root}>
      <Grid container direction="column" justify="center" alignItems="center">
        <Grid className={classes.title} item xs={12}>
          <Typography variant="h6">
            Choose your strategy
          </Typography>
        </Grid>
        <Grid className={classes.gridItem} item xs={12}>
          <InvestmentStrategyRadios />
        </Grid>
        <Grid className={classes.title} item xs={12}>
          <Typography variant="h6">
            Select Stocks
          </Typography>
        </Grid>
        <Grid className={classes.gridItem} item xs={12} >
          <StockSelectSection selectedStocks={selectedStocks}>
          </StockSelectSection>
        </Grid>
        <Grid className={classes.gridItem} item xs={12} >
          <SubmitSelectionButton />
        </Grid>
      </Grid>
    </div>
  );
}

export default PortfolioPage;

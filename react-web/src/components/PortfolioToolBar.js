import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import StockSearchBox from '../components/StockSearchBox'
import SelectPortfolioButton from '../components/SelectPortfolioButton';

const useStyles = makeStyles((theme) => ({
  root: {
    width: '100%',
    minHeight: '60px',
    padding: theme.spacing(1, 2, 1),
    flexGrow: 1,
    boxShadow: '0 0 1px 1px rgba(0, 0, 0 ,.1)'
  },
}));

export default function PortfolioToolBar(props) {
  const classes = useStyles();

  return (
    <Grid
      className={classes.root}
      container
      direction="row"
      justify="flex-start"
      alignItems="center"
    >
      <Grid item xs={12} sm={3}>
        <SelectPortfolioButton >
        </SelectPortfolioButton>
      </Grid>
      <Grid item container xs={12} sm={6} justify = "center" >
        <StockSearchBox
            selectedStocks={props.selectedStocks} 
            setSelectedStocks={props.setSelectedStocks}
            companyData={props.companyData}
            setDialogMessage={props.setDialogMessage}
            openMessageDialog={props.openMessageDialog}
          >
        </StockSearchBox>
      </Grid>
    </Grid>
  );
}
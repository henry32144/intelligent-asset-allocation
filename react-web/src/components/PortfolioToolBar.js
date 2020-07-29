import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import StockSearchBox from '../components/StockSearchBox'
import PortfolioMenuButtons from './PortfolioMenuButtons';
import PortfolioDetailButtons from './PortfolioDetailButtons';
import Hidden from '@material-ui/core/Hidden';

const useStyles = makeStyles((theme) => ({
  root: {
    width: '100%',
    minHeight: '60px',
    padding: theme.spacing(1, 2, 1),
    boxShadow: '0 0 1px 1px rgba(0, 0, 0 ,.1)'
  },
  menuButtons: {
    display: 'inline-flex'
  }
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
      <Grid item xs={12} sm={3} className={classes.menuButtons}>
        <PortfolioMenuButtons
          isSideBarExpanded={props.isSideBarExpanded}
          setSideBarExpand={props.setSideBarExpand}
          setSelectedStocks={props.setSelectedStocks}
          userPortfolios={props.userPortfolios}
          setUserPortfolios={props.setUserPortfolios}
          currentSelectedPortfolio={props.currentSelectedPortfolio}
          setCurrentSelectedPortfolio={props.setCurrentSelectedPortfolio}
          handleCreatePortfolioDialogOpen={props.handleCreatePortfolioDialogOpen}
          setDialogTitle={props.setDialogTitle}
          setDialogMessage={props.setDialogMessage}
          openMessageDialog={props.openMessageDialog}
          userData={props.userData}
        >
        </PortfolioMenuButtons>
        <Hidden smUp>
          <PortfolioDetailButtons
            showSearchButton={true}
          >
          </PortfolioDetailButtons>
        </Hidden>
      </Grid>
      <Grid item container xs={12} sm={6} justify = "center" >
        <StockSearchBox
            selectedStocks={props.selectedStocks} 
            setSelectedStocks={props.setSelectedStocks}
            companyData={props.companyData}
            setDialogTitle={props.setDialogTitle}
            setDialogMessage={props.setDialogMessage}
            openMessageDialog={props.openMessageDialog}
          >
        </StockSearchBox>
      </Grid>
      <Hidden xsDown>
        <Grid item container sm={3} justify = "center" >
          <PortfolioDetailButtons>
          </PortfolioDetailButtons>
        </Grid>
      </Hidden>
    </Grid>
  );
}
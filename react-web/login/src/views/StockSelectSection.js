import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import StockSelectedList from '../components/StockSelectedList'

const useStyles = makeStyles((theme) => ({
  root: {
    height: '100vh',
    padding: theme.spacing(1, 1, 1),
    flexGrow: 1,
    [theme.breakpoints.up('xs')]: {
      width: '60vw',
    },
    [theme.breakpoints.up('md')]: {
      width: '50vw',
    },
    [theme.breakpoints.up('lg')]: {
      width: '25vw',
    },
    position: "fixed",
    zIndex: 1400,
    background: 'white'
  },
}));

export default function StockSelectSection(props) {
  const classes = useStyles();

  return (
    <Box className={classes.root} boxShadow={1}>
      <StockSelectedList
        selectedStocks={props.selectedStocks}
        setSelectedStocks={props.setSelectedStocks}
        currentSelectedStock={props.currentSelectedStock}
        setCurrentSelectedStock={props.setCurrentSelectedStock}
        savePortfolio={props.savePortfolio}
        saveButtonLoading={props.saveButtonLoading}
      />
    </Box>
  );
}
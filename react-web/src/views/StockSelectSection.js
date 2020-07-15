import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import StockSelectedList from '../components/StockSelectedList'
import StockSearchBox from '../components/StockSearchBox'

const useStyles = makeStyles((theme) => ({
  root: {
    padding: theme.spacing(2, 2, 2),
    flexGrow: 1,
  },
  stockComponent: {
      margin: theme.spacing(0, 0, 2),
  }
}));
  
export default function StockSelectSection() {
  const classes = useStyles();

  return (
    <Box className={classes.root} boxShadow={0}>
      <StockSearchBox additionalStyles={classes} />
      <StockSelectedList additionalStyles={classes} />
    </Box>
  );
}